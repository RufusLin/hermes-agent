"""TLS trust contract: the OS certificate store, with explicit config on top.

These are behavior contracts, not snapshots — they assert WHERE trust comes
from and that explicit per-provider settings still beat it.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

import pytest

from agent.ssl_verify import resolve_httpx_verify
from datetime import UTC


@pytest.fixture
def no_ca_env(monkeypatch):
    """Exported CA env vars flip resolve_httpx_verify() from ``True`` to a
    shared union context (or fail closed if bad). Pin the env for fallback
    assertions rather than inheriting a developer shell's values."""
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)


def test_missing_explicit_bundle_falls_back_to_the_platform_store(tmp_path, caplog, no_ca_env):
    missing = str(tmp_path / "nope.pem")

    assert resolve_httpx_verify(ca_bundle=missing) is True
    assert "does not exist" in caplog.text


def test_env_ca_bundle_shares_union_context_and_missing_fails_closed(tmp_path, monkeypatch, no_ca_env):
    """With SSL_CERT_FILE exported to a readable bundle, resolve_httpx_verify()
    shares the same union context as platform_ssl_context() (bundle + OS store);
    a missing per-provider ca_bundle falls back to that same union context, while
    a missing SSL_CERT_FILE fails closed."""
    import ssl

    import certifi

    from agent import ssl_verify

    monkeypatch.setattr(ssl_verify, "_CA_CONTEXTS", {})
    monkeypatch.setenv("SSL_CERT_FILE", certifi.where())
    union_ctx = resolve_httpx_verify()
    assert union_ctx is not True and union_ctx is not False
    assert resolve_httpx_verify(ca_bundle=str(tmp_path / "nope.pem")) is union_ctx

    bundle_ctx = ssl_verify.platform_ssl_context()
    assert bundle_ctx is ssl_verify.platform_ssl_context()
    assert bundle_ctx is union_ctx
    assert type(bundle_ctx) is ssl.SSLContext  # truststore-injected: bundle + OS store
    assert bundle_ctx is ssl_verify._CA_CONTEXTS[(str(Path(certifi.where()).resolve()), True)]

    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "missing-ca.pem"))
    with pytest.raises(FileNotFoundError):
        resolve_httpx_verify()


@pytest.mark.parametrize("value", [False, "false", "0", "no", "off", "FALSE"])
def test_insecure_disables_verification(value):
    assert resolve_httpx_verify(ssl_verify=value) is False


def test_insecure_beats_an_explicit_bundle():
    import certifi

    assert resolve_httpx_verify(ca_bundle=certifi.where(), ssl_verify=False) is False


def test_truststore_failure_degrades_to_openssl_defaults():
    import subprocess
    import sys

    child = subprocess.run([sys.executable, "-c", """
import builtins, ssl
original = ssl.SSLContext
real_import = builtins.__import__
def no_truststore(name, *args, **kwargs):
    if name == 'truststore':
        raise ImportError('unavailable fixture')
    return real_import(name, *args, **kwargs)
builtins.__import__ = no_truststore
from agent.ssl_verify import install_truststore, resolve_httpx_verify
assert install_truststore() is False
assert install_truststore() is False
assert ssl.SSLContext is original
import httpx
with httpx.Client(verify=resolve_httpx_verify()) as client:
    ctx = client._transport._pool._ssl_context
    assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname
"""], capture_output=True, text=True, timeout=30, check=False)
    assert child.returncode == 0, child.stderr
    assert "truststore unavailable" in child.stderr


@contextmanager
def _private_https_server(tmp_path: Path, *, common_name: str = "Private provider") -> Iterator[tuple[Any, Path]]:
    """A TLS server no public certificate store has ever seen.

    Yields ``(server, ca)``: the running server, and the PEM of the CA that signed
    its certificate — the only bundle that can trust it.
    """
    from datetime import datetime, timedelta
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from ipaddress import ip_address
    import ssl
    import threading

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])
    now = datetime.now(UTC)
    cert = (x509.CertificateBuilder().subject_name(subject).issuer_name(subject)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(days=1)).not_valid_after(now + timedelta(days=1))
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ip_address("127.0.0.1"))]), critical=False)
            # macOS Security.framework rejects a server cert without these ("not permitted
            # for this usage"); OpenSSL-backed platforms accept it either way.
            .add_extension(x509.KeyUsage(digital_signature=True, key_encipherment=True, key_cert_sign=True,
                                         content_commitment=False, data_encipherment=False, key_agreement=False,
                                         crl_sign=False, encipher_only=False, decipher_only=False), critical=True)
            .add_extension(x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .sign(key, hashes.SHA256()))
    ca = tmp_path / "provider-ca.pem"
    private = tmp_path / "provider.key"
    ca.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    private.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                          serialization.NoEncryption()))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"private provider")

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    # Earlier tests may have injected truststore; its wrapper is client-only.
    from agent.ssl_verify import _stdlib_ssl_context_class
    context = _stdlib_ssl_context_class()(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(ca, private)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, ca
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _child_env(tmp_path: Path) -> dict[str, str]:
    """A subprocess environment with no inherited TLS overrides."""
    import os

    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(HOME=str(tmp_path), HERMES_HOME=str(home), NO_PROXY="127.0.0.1")
    for key in ("SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        env.pop(key, None)
    return env


def test_explicit_provider_ca_replaces_platform_trust_on_real_https(tmp_path):
    """A private endpoint trusts only its provider CA, never a global fallback."""
    import subprocess
    import sys

    with _private_https_server(tmp_path) as (server, ca):
        script = """
import httpx, sys
from agent.ssl_verify import resolve_httpx_verify
url, ca = sys.argv[1:]
try:
    httpx.get(url, verify=resolve_httpx_verify(), timeout=5)
except httpx.ConnectError:
    pass
else:
    raise AssertionError('unconfigured platform store trusted private provider')
response = httpx.get(url, verify=resolve_httpx_verify(ca_bundle=ca), timeout=5)
assert response.content == b'private provider', response
"""
        child = subprocess.run(
            [sys.executable, "-c", script, f"https://127.0.0.1:{server.server_port}/", str(ca)],
            env=_child_env(tmp_path), capture_output=True, text=True, timeout=30, check=False,
        )
        assert child.returncode == 0, child.stdout + child.stderr


def test_explicit_bundle_works_without_truststore():
    """A provider ``ssl_ca_cert`` on an interpreter without truststore
    (3.11–3.13 bridge installs) must still yield a verifying context built
    on that bundle, not an import error at client construction."""
    import subprocess
    import sys

    child = subprocess.run([sys.executable, "-c", """
import sys, ssl, certifi
sys.modules['truststore'] = None
from agent.ssl_verify import resolve_httpx_verify
ctx = resolve_httpx_verify(ca_bundle=certifi.where())
assert isinstance(ctx, ssl.SSLContext), ctx
assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname
assert ctx.cert_store_stats()['x509_ca'] > 0
"""], capture_output=True, text=True, timeout=30, check=False)
    assert child.returncode == 0, child.stderr


# ---------------------------------------------------------------------------
# An install environment replaced under a live process: certifi's path dies
# ---------------------------------------------------------------------------


class _Rotation(NamedTuple):
    """One install-environment rotation, as the live process sees it."""

    module: Any
    """The certifi imported from the environment that was then deleted."""
    stale_package: Path
    """``env-a/certifi`` — gone from disk, still the path certifi reports."""
    live_package: Path | None
    """``env-b/certifi`` when a replacement was installed, else ``None``."""


def _stand_in_certifi_env(root: Path, name: str, bundle: bytes) -> Path:
    """A byte-for-byte copy of the INSTALLED certifi package.

    The stand-in has to be the real package, not a file that recomputes its own path
    on every call: certifi's ``where`` lives in ``certifi.core`` and caches the
    resolved path in a module global, and clearing that caching is exactly what a
    repair has to do. A single-file stand-in passes while the real package stays
    broken.
    """
    import shutil

    import certifi as installed

    package = root / name / "certifi"
    shutil.copytree(Path(installed.__file__).parent, package,
                    ignore=shutil.ignore_patterns("__pycache__"))
    (package / "cacert.pem").write_bytes(bundle)  # a real PEM: contexts must load it
    return package


def _client_ssl_context(client):
    """The context httpx built for this client's connection pool."""
    from typing import cast

    return cast(Any, client._transport)._pool._ssl_context


def _rotated_certifi(tmp_path: Path, monkeypatch, *, replacement: bool) -> _Rotation:
    """Delete the environment directory the imported certifi points into.

    The package manager's rotation without a package manager: certifi is imported
    from ``env-a``, then that directory is removed. ``replacement`` adds the
    identical package the manager would have installed as ``env-b``.
    """
    import importlib
    import shutil
    import sys

    import certifi as real_certifi
    import requests.utils

    bundle = Path(real_certifi.where()).read_bytes()  # a real PEM: contexts must load it
    stale = _stand_in_certifi_env(tmp_path, "env-a", bundle)
    live = None
    if replacement:
        live = _stand_in_certifi_env(tmp_path, "env-b", bundle)
        monkeypatch.syspath_prepend(str(live.parent))
    monkeypatch.syspath_prepend(str(stale.parent))
    for name in [n for n in sys.modules if n == "certifi" or n.startswith("certifi.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    importlib.invalidate_caches()
    stand_in = importlib.import_module("certifi")
    monkeypatch.setitem(sys.modules, "certifi", stand_in)
    assert Path(stand_in.where()).parent == stale
    # requests froze its default bundle path from the same certifi at import.
    monkeypatch.setattr(requests.utils, "DEFAULT_CA_BUNDLE_PATH", stand_in.where())
    shutil.rmtree(stale)
    assert not Path(stand_in.where()).exists()
    return _Rotation(stand_in, stale, live)


def _ssl_verify_warnings(caplog) -> list:
    return [record for record in caplog.records if record.name == "agent.ssl_verify"]


def test_replaced_install_environment_re_resolves_certifi_in_place(tmp_path, monkeypatch, caplog, no_ca_env):
    """A process that outlives an environment rotation holds certifi's path into the
    deleted directory, and httpx builds every SSL context through it. The package is
    re-imported against the live ``sys.path``, so client construction succeeds again
    for this process and every other certifi consumer in it."""
    import logging
    import ssl
    import sys

    import httpx

    rotation = _rotated_certifi(tmp_path, monkeypatch, replacement=True)
    stale_core = sys.modules["certifi.core"]

    with caplog.at_level(logging.WARNING, logger="agent.ssl_verify"):
        verify = resolve_httpx_verify()

    assert verify is True  # httpx's own default, now backed by a bundle that exists
    with httpx.Client(verify=verify) as client:  # the construction that raised FileNotFoundError
        assert _client_ssl_context(client).verify_mode == ssl.CERT_REQUIRED
    # The repair reaches certifi itself, not just this call's return value: httpx
    # resolves verify=True through certifi.where() at every construction.
    assert rotation.live_package is not None
    live_bundle = Path(sys.modules["certifi"].where())
    assert sys.modules["certifi"] is not rotation.module
    assert sys.modules["certifi.core"] is not stale_core
    assert live_bundle.is_file() and live_bundle.parent.resolve() == rotation.live_package.resolve()
    live_core_file = Path(str(getattr(sys.modules["certifi.core"], "__file__", "")))
    assert live_core_file.parent.resolve() == rotation.live_package.resolve()
    logged = _ssl_verify_warnings(caplog)
    assert len(logged) == 1 and "re-resolved" in logged[0].getMessage()
    assert str(rotation.stale_package) in logged[0].getMessage()


def test_rotation_repair_and_warning_happen_once(tmp_path, monkeypatch, caplog, no_ca_env):
    """A gateway builds a TLS client per turn, per cron run and per model-metadata
    fetch: the repair re-executes third-party module code, so it — and the single
    warning an operator needs — must not repeat per client."""
    import logging

    from agent import ssl_verify

    _rotated_certifi(tmp_path, monkeypatch, replacement=True)
    reloads: list[int] = []
    real_reload = ssl_verify._reload_certifi_bundle

    def counting_reload():
        reloads.append(1)
        return real_reload()

    monkeypatch.setattr(ssl_verify, "_reload_certifi_bundle", counting_reload)

    with caplog.at_level(logging.WARNING, logger="agent.ssl_verify"):
        for _ in range(5):
            assert ssl_verify.resolve_httpx_verify() is True
            ssl_verify.requests_verify()

    assert len(reloads) == 1
    assert len(_ssl_verify_warnings(caplog)) == 1


def test_unrecoverable_bundle_hands_over_the_platform_context(tmp_path, monkeypatch, caplog, no_ca_env):
    """Nothing left to re-resolve: the platform store is handed over instead of
    raising, which is the difference between a degraded process and a fleet-wide
    TLS outage that only a restart clears."""
    import logging
    import ssl

    import httpx

    from agent import ssl_verify

    _rotated_certifi(tmp_path, monkeypatch, replacement=False)
    attempts: list[int] = []

    def unrecoverable() -> None:
        attempts.append(1)  # the answer this stand-in gives is None

    monkeypatch.setattr(ssl_verify, "_reload_certifi_bundle", unrecoverable)

    verify: Any = None
    with caplog.at_level(logging.WARNING, logger="agent.ssl_verify"):
        for _ in range(5):
            verify = resolve_httpx_verify()

    assert isinstance(verify, ssl.SSLContext)
    assert len(attempts) == 1  # a dead path is not re-probed per client
    logged = _ssl_verify_warnings(caplog)
    assert len(logged) == 1 and "could not be re-resolved" in logged[0].getMessage()
    with httpx.Client(verify=verify) as client:
        assert _client_ssl_context(client).verify_mode == ssl.CERT_REQUIRED


def test_requests_verify_hands_over_a_bundle_that_exists(tmp_path, monkeypatch, no_ca_env):
    """``requests`` freezes its default bundle path when it is imported, so no reload
    can reach it: its call sites ask for a live bundle instead."""
    from agent import ssl_verify

    rotation = _rotated_certifi(tmp_path, monkeypatch, replacement=True)

    verify = ssl_verify.requests_verify()

    assert rotation.live_package is not None
    assert Path(str(verify)).resolve() == (rotation.live_package / "cacert.pem").resolve()
    assert Path(str(verify)).is_file()


def test_requests_verify_never_hands_over_the_deleted_bundle(tmp_path, monkeypatch, no_ca_env):
    """With nothing re-resolvable the answer is OpenSSL's own bundle, or requests'
    own default — never the path the rotation deleted."""
    from agent import ssl_verify

    rotation = _rotated_certifi(tmp_path, monkeypatch, replacement=False)
    dead = rotation.module.where()
    monkeypatch.setattr(ssl_verify, "_reload_certifi_bundle", lambda: None)

    verify = ssl_verify.requests_verify()

    assert str(verify) != dead
    assert verify is True or Path(str(verify)).exists()


def test_rotated_process_keeps_trusting_the_bundle_it_was_configured_with(tmp_path):
    """End to end over real TLS: a process whose install environment was replaced
    still builds a client that trusts the CA it was configured with.

    The private server is trusted by the rotated bundle and by nothing else, so a
    process that fell back to the platform store — or that kept pointing into the
    deleted directory — cannot complete this request.
    """
    import subprocess
    import sys

    with _private_https_server(tmp_path) as (server, ca):
        bundle = ca.read_bytes()
        stale = _stand_in_certifi_env(tmp_path, "env-a", bundle)
        live = _stand_in_certifi_env(tmp_path, "env-b", bundle)
        script = """
import shutil, sys
from pathlib import Path
url, env_a, env_b = sys.argv[1:4]
sys.path.insert(0, env_a)
import certifi
assert Path(certifi.where()).parent.resolve() == Path(env_a, 'certifi').resolve(), certifi.where()
sys.path.insert(0, env_b)
sys.path.remove(env_a)
shutil.rmtree(env_a)
import httpx
from agent.ssl_verify import resolve_httpx_verify
response = httpx.get(url, verify=resolve_httpx_verify(), timeout=5)
assert response.content == b'private provider', response.content
live = Path(sys.modules['certifi'].where())
assert live.is_file() and live.parent.resolve() == Path(env_b, 'certifi').resolve(), live
"""
        child = subprocess.run(
            [sys.executable, "-c", script, f"https://127.0.0.1:{server.server_port}/",
             str(stale.parent), str(live.parent)],
            env=_child_env(tmp_path), capture_output=True, text=True, timeout=60, check=False,
        )
        assert child.returncode == 0, child.stdout + child.stderr


def test_requests_verify_keeps_requests_own_ca_overrides(monkeypatch, no_ca_env):
    """requests reads ``REQUESTS_CA_BUNDLE`` only for ``verify=True``; a healthy install
    or a configured override must keep it, or a proxy CA stops being trusted."""
    from agent import ssl_verify

    assert ssl_verify.requests_verify() is True
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", "/corp/proxy-ca.pem")
    monkeypatch.setattr("requests.utils.DEFAULT_CA_BUNDLE_PATH", "/gone/cacert.pem")
    assert ssl_verify.requests_verify() is True



def test_requests_verify_hands_over_the_platform_store_without_openssl_files(tmp_path, monkeypatch, no_ca_env):
    """Windows: requests' frozen bundle is gone, certifi cannot be re-derived and OpenSSL has no
    cafile/capath. ``True`` would make requests raise on the dead path, so the OS store goes over
    as a PEM bundle that requests accepts."""
    import ssl
    from types import SimpleNamespace

    import requests

    from agent import ssl_verify

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
    monkeypatch.delenv("CURL_CA_BUNDLE", raising=False)
    monkeypatch.setattr(ssl_verify, "_PLATFORM_STORE_BUNDLE", None)
    monkeypatch.setattr(ssl_verify, "certifi_bundle_path", lambda: None)
    monkeypatch.setattr("requests.utils.DEFAULT_CA_BUNDLE_PATH", str(tmp_path / "gone" / "cacert.pem"))
    monkeypatch.setattr(ssl, "get_default_verify_paths", lambda: SimpleNamespace(cafile=None, capath=None))
    # Windows enumerates its system store into the context; a Linux CA directory loads lazily and
    # lists nothing. Stand in a store holding one real root so the test means the same everywhere.
    import certifi

    pem = Path(certifi.where()).read_text().split("-----END CERTIFICATE-----")[0] + "-----END CERTIFICATE-----\n"
    root_der = ssl.PEM_cert_to_DER_cert(pem[pem.index("-----BEGIN CERTIFICATE-----"):])

    class _WindowsLikeStore(ssl.SSLContext):
        def get_ca_certs(self, binary_form=False):
            return [root_der]

    monkeypatch.setattr(ssl_verify, "_stdlib_ssl_context_class", lambda: _WindowsLikeStore)

    verify = ssl_verify.requests_verify()
    assert isinstance(verify, str) and Path(verify).is_relative_to(tmp_path / "home")
    assert "BEGIN CERTIFICATE" in Path(verify).read_text()
    # requests takes the bundle: the request fails on the closed port, not on a missing CA file.
    with pytest.raises(requests.exceptions.ConnectionError):
        requests.get("https://127.0.0.1:9/", verify=verify, timeout=2)


def test_platform_store_bundle_survives_an_unwritable_home(tmp_path, monkeypatch):
    """A full or read-only HERMES_HOME leaves requests_verify on its ``True`` fallback
    instead of raising the disk error out of the requests call site."""
    from agent import ssl_verify

    import ssl

    class _StoreWithACert(ssl.SSLContext):
        # A Linux CA directory loads lazily and lists nothing; give the store one cert so the
        # write (and its failure) is what decides the result.
        def get_ca_certs(self, binary_form=False):
            return [b"0"]

    blocker = tmp_path / "home"
    blocker.write_text("not a directory")  # mkdir(cache) under a file raises OSError
    monkeypatch.setenv("HERMES_HOME", str(blocker))
    monkeypatch.setattr(ssl_verify, "_PLATFORM_STORE_BUNDLE", None)
    monkeypatch.setattr(ssl_verify, "_stdlib_ssl_context_class", lambda: _StoreWithACert)
    assert ssl_verify._platform_store_bundle() is None


# ---------------------------------------------------------------------------
# resolve_httpx_verify() env CA union & fail-closed contracts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env_name", ["SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"])
def test_resolve_httpx_verify_missing_env_ca_file_fails_closed(tmp_path, monkeypatch, no_ca_env, env_name):
    """A missing CA file in SSL_CERT_FILE or REQUESTS_CA_BUNDLE must raise, never fall back or return False."""
    from agent import ssl_verify

    monkeypatch.setattr(ssl_verify, "_CA_CONTEXTS", {})
    monkeypatch.setenv(env_name, str(tmp_path / "does-not-exist.pem"))
    with pytest.raises(FileNotFoundError):
        resolve_httpx_verify()


@pytest.mark.parametrize("env_name", ["SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"])
def test_resolve_httpx_verify_non_file_and_invalid_bundle_fail_closed(tmp_path, monkeypatch, no_ca_env, env_name):
    """Directory, empty file, and non-cert text in SSL_CERT_FILE / REQUESTS_CA_BUNDLE must raise."""
    import ssl

    from agent import ssl_verify

    monkeypatch.setattr(ssl_verify, "_CA_CONTEXTS", {})
    ca_dir = tmp_path / "dir"
    ca_dir.mkdir()
    monkeypatch.setenv(env_name, str(ca_dir))
    with pytest.raises(ValueError):
        resolve_httpx_verify()

    empty_file = tmp_path / "empty.pem"
    empty_file.write_text("")
    monkeypatch.setenv(env_name, str(empty_file))
    with pytest.raises(ssl.SSLError):
        resolve_httpx_verify()

    bad_file = tmp_path / "not-a-cert.pem"
    bad_file.write_text("not a valid certificate\n")
    monkeypatch.setenv(env_name, str(bad_file))
    with pytest.raises(ssl.SSLError):
        resolve_httpx_verify()


@pytest.mark.platforms("posix")
def test_resolve_httpx_verify_unreadable_env_ca_file_fails_closed(tmp_path, monkeypatch, no_ca_env):
    """An unreadable SSL_CERT_FILE fails closed with PermissionError."""
    import os

    import certifi

    from agent import ssl_verify

    if os.geteuid() == 0:
        pytest.skip("root bypasses file mode 000")

    monkeypatch.setattr(ssl_verify, "_CA_CONTEXTS", {})
    unreadable = tmp_path / "unreadable.pem"
    unreadable.write_text(Path(certifi.where()).read_text())
    unreadable.chmod(0o000)
    try:
        monkeypatch.setenv("SSL_CERT_FILE", str(unreadable))
        with pytest.raises(PermissionError):
            resolve_httpx_verify()
    finally:
        unreadable.chmod(0o600)


def test_resolve_httpx_verify_bad_ssl_cert_file_does_not_fall_back_to_requests_ca_bundle(
    tmp_path, monkeypatch, no_ca_env
):
    """If SSL_CERT_FILE is set to a bad path, fail closed even if REQUESTS_CA_BUNDLE is valid."""
    import certifi

    from agent import ssl_verify

    monkeypatch.setattr(ssl_verify, "_CA_CONTEXTS", {})
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "missing.pem"))
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", certifi.where())
    with pytest.raises(FileNotFoundError):
        resolve_httpx_verify()


def test_resolve_httpx_verify_ssl_cert_dir_missing_or_file_fails_closed(tmp_path, monkeypatch, no_ca_env):
    """SSL_CERT_DIR without a file env fails closed when missing or pointing to a regular file."""
    from agent import ssl_verify

    monkeypatch.setattr(ssl_verify, "_CA_CONTEXTS", {})
    monkeypatch.setenv("SSL_CERT_DIR", str(tmp_path / "no-such-dir"))
    with pytest.raises(FileNotFoundError):
        resolve_httpx_verify()

    not_a_dir = tmp_path / "file.pem"
    not_a_dir.write_text("x")
    monkeypatch.setenv("SSL_CERT_DIR", str(not_a_dir))
    with pytest.raises(NotADirectoryError):
        resolve_httpx_verify()


@pytest.mark.parametrize("env_name", ["SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"])
def test_resolve_httpx_verify_file_and_bad_ssl_cert_dir_fails_closed(
    tmp_path, monkeypatch, no_ca_env, env_name
):
    """Valid SSL_CERT_FILE / REQUESTS_CA_BUNDLE with missing or non-dir SSL_CERT_DIR must fail closed."""
    import certifi

    from agent import ssl_verify

    monkeypatch.setattr(ssl_verify, "_CA_CONTEXTS", {})
    monkeypatch.setenv(env_name, certifi.where())
    monkeypatch.setenv("SSL_CERT_DIR", str(tmp_path / "missing-cert-dir"))
    with pytest.raises(FileNotFoundError):
        resolve_httpx_verify()

    not_a_dir = tmp_path / "not-a-dir.pem"
    not_a_dir.write_text("x")
    monkeypatch.setenv("SSL_CERT_DIR", str(not_a_dir))
    with pytest.raises(NotADirectoryError):
        resolve_httpx_verify()


@pytest.mark.platforms("posix")
def test_resolve_httpx_verify_file_and_unreadable_ssl_cert_dir_fails_closed(
    tmp_path, monkeypatch, no_ca_env
):
    """Valid SSL_CERT_FILE with an unreadable SSL_CERT_DIR fails closed with PermissionError."""
    import os

    import certifi

    from agent import ssl_verify

    if os.geteuid() == 0:
        pytest.skip("root bypasses directory mode 000")

    monkeypatch.setattr(ssl_verify, "_CA_CONTEXTS", {})
    unreadable_dir = tmp_path / "unreadable-certs"
    unreadable_dir.mkdir()
    unreadable_dir.chmod(0o000)
    try:
        monkeypatch.setenv("SSL_CERT_FILE", certifi.where())
        monkeypatch.setenv("SSL_CERT_DIR", str(unreadable_dir))
        with pytest.raises(PermissionError):
            resolve_httpx_verify()
    finally:
        unreadable_dir.chmod(0o700)


def test_resolve_httpx_verify_whitespace_ssl_cert_dir_treated_as_unset(tmp_path, monkeypatch, no_ca_env):
    """Whitespace-only SSL_CERT_DIR is treated as unset with or without a valid file env."""
    import certifi

    from agent import ssl_verify

    monkeypatch.setattr(ssl_verify, "_CA_CONTEXTS", {})
    baseline_no_env = resolve_httpx_verify()
    monkeypatch.setenv("SSL_CERT_DIR", "   \t\n ")
    assert resolve_httpx_verify() is baseline_no_env

    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    monkeypatch.setenv("SSL_CERT_FILE", certifi.where())
    file_only_ctx = resolve_httpx_verify()

    monkeypatch.setenv("SSL_CERT_DIR", "   ")
    assert resolve_httpx_verify() is file_only_ctx


def test_resolve_httpx_verify_empty_or_unhashed_ssl_cert_dir_treated_as_unset(
    tmp_path, monkeypatch, no_ca_env
):
    """Policy A: an existing, readable SSL_CERT_DIR with no <hash>.<n> entries is treated as unset."""
    import certifi

    from agent import ssl_verify

    monkeypatch.setattr(ssl_verify, "_CA_CONTEXTS", {})
    empty_dir = tmp_path / "empty-certs"
    empty_dir.mkdir()
    unhashed_dir = tmp_path / "unhashed-certs"
    unhashed_dir.mkdir()
    (unhashed_dir / "ca.pem").write_text(Path(certifi.where()).read_text())

    baseline_no_env = resolve_httpx_verify()
    for candidate in (empty_dir, unhashed_dir):
        monkeypatch.setenv("SSL_CERT_DIR", str(candidate))
        assert resolve_httpx_verify() is baseline_no_env

    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    monkeypatch.setenv("SSL_CERT_FILE", certifi.where())
    file_only_ctx = resolve_httpx_verify()
    for candidate in (empty_dir, unhashed_dir):
        monkeypatch.setenv("SSL_CERT_DIR", str(candidate))
        assert resolve_httpx_verify() is file_only_ctx


def test_resolve_httpx_verify_both_file_envs_valid_ssl_cert_file_wins(
    tmp_path, monkeypatch, no_ca_env
):
    """When both SSL_CERT_FILE and REQUESTS_CA_BUNDLE are valid, SSL_CERT_FILE wins."""
    import certifi

    from agent import ssl_verify

    monkeypatch.setattr(ssl_verify, "_CA_CONTEXTS", {})
    alt_bundle = tmp_path / "alt-bundle.pem"
    alt_bundle.write_text(Path(certifi.where()).read_text())

    monkeypatch.setenv("SSL_CERT_FILE", certifi.where())
    ssl_file_only = resolve_httpx_verify()

    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(alt_bundle))
    assert resolve_httpx_verify() is ssl_file_only


def test_resolve_httpx_verify_ssl_cert_dir_trusts_hashed_ca_and_rejects_untrusted(tmp_path):
    """With only SSL_CERT_DIR set to a hashed CA directory, resolve_httpx_verify()
    trusts that CA via a truststore union context while rejecting an untrusted CA."""
    import shutil
    import subprocess
    import sys

    openssl = shutil.which("openssl")
    if not openssl:
        pytest.skip("openssl CLI required to compute subject hash for SSL_CERT_DIR")

    trusted_dir = tmp_path / "trusted"
    untrusted_dir = tmp_path / "untrusted"
    ca_dir = tmp_path / "certs"
    trusted_dir.mkdir()
    untrusted_dir.mkdir()
    ca_dir.mkdir()

    with _private_https_server(trusted_dir) as (trusted_srv, trusted_ca), \
         _private_https_server(untrusted_dir) as (untrusted_srv, _untrusted_ca):
        cert_hash = subprocess.check_output(
            [openssl, "x509", "-hash", "-noout", "-in", str(trusted_ca)],
            text=True,
        ).strip()
        (ca_dir / f"{cert_hash}.0").symlink_to(trusted_ca)
        script = """
import httpx, ssl, sys
from agent.ssl_verify import resolve_httpx_verify
trusted_url, untrusted_url = sys.argv[1:3]
verify = resolve_httpx_verify()
assert isinstance(verify, ssl.SSLContext), verify
assert type(verify).__module__.startswith("truststore"), type(verify)
resp = httpx.get(trusted_url, verify=verify, timeout=5)
assert resp.content == b"private provider", resp.content
try:
    httpx.get(untrusted_url, verify=verify, timeout=5)
except httpx.ConnectError:
    pass
else:
    raise AssertionError("untrusted server cert was accepted")
"""
        env = _child_env(tmp_path)
        env["SSL_CERT_DIR"] = str(ca_dir)
        child = subprocess.run(
            [
                sys.executable, "-c", script,
                f"https://127.0.0.1:{trusted_srv.server_port}/",
                f"https://127.0.0.1:{untrusted_srv.server_port}/",
            ],
            env=env, capture_output=True, text=True, timeout=30, check=False,
        )
        assert child.returncode == 0, child.stdout + child.stderr


def test_resolve_httpx_verify_unset_env_matches_default_behaviour(tmp_path, monkeypatch, no_ca_env):
    """With no CA env vars set, resolve_httpx_verify() keeps its pre-change _default_verify() behaviour."""
    from agent import ssl_verify

    assert resolve_httpx_verify() is True

    _rotated_certifi(tmp_path, monkeypatch, replacement=False)
    monkeypatch.setattr(ssl_verify, "_reload_certifi_bundle", lambda: None)
    assert resolve_httpx_verify() is ssl_verify._shared_context(None)


@pytest.mark.parametrize("env_name", ["SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"])
def test_resolve_httpx_verify_env_ca_trusts_bundle_and_rejects_untrusted_on_real_https(tmp_path, env_name):
    """With SSL_CERT_FILE (or REQUESTS_CA_BUNDLE fallback) set, resolve_httpx_verify() trusts
    that bundle via a truststore union context while rejecting an untrusted server cert."""
    import subprocess
    import sys

    trusted_dir = tmp_path / "trusted"
    untrusted_dir = tmp_path / "untrusted"
    trusted_dir.mkdir()
    untrusted_dir.mkdir()

    with _private_https_server(trusted_dir) as (trusted_srv, trusted_ca), \
         _private_https_server(untrusted_dir) as (untrusted_srv, _untrusted_ca):
        script = """
import httpx, ssl, sys
from agent.ssl_verify import resolve_httpx_verify
trusted_url, untrusted_url = sys.argv[1:3]
verify = resolve_httpx_verify()
assert isinstance(verify, ssl.SSLContext), verify
assert type(verify).__module__.startswith("truststore"), type(verify)
resp = httpx.get(trusted_url, verify=verify, timeout=5)
assert resp.content == b"private provider", resp.content
try:
    httpx.get(untrusted_url, verify=verify, timeout=5)
except httpx.ConnectError:
    pass
else:
    raise AssertionError("untrusted server cert was accepted")
"""
        env = _child_env(tmp_path)
        env[env_name] = str(trusted_ca)
        child = subprocess.run(
            [
                sys.executable, "-c", script,
                f"https://127.0.0.1:{trusted_srv.server_port}/",
                f"https://127.0.0.1:{untrusted_srv.server_port}/",
            ],
            env=env, capture_output=True, text=True, timeout=30, check=False,
        )
        assert child.returncode == 0, child.stdout + child.stderr


def test_resolve_httpx_verify_unions_env_bundle_with_platform_store_without_replacing_it(tmp_path):
    """The context returned for SSL_CERT_FILE unions the exported CA with the platform store
    rather than replacing it: a server signed by the exported CA and a server signed only by
    the platform store both verify, while an unrelated CA is still rejected."""
    import subprocess
    import sys

    env_dir = tmp_path / "env_ca"
    platform_dir = tmp_path / "platform_ca"
    untrusted_dir = tmp_path / "untrusted_ca"
    env_dir.mkdir()
    platform_dir.mkdir()
    untrusted_dir.mkdir()

    with _private_https_server(env_dir, common_name="Env CA") as (env_srv, env_ca), \
         _private_https_server(platform_dir, common_name="Platform CA") as (plat_srv, plat_ca), \
         _private_https_server(untrusted_dir, common_name="Untrusted CA") as (untrusted_srv, _untrusted_ca):
        script = """
import httpx, ssl, sys
env_url, plat_url, untrusted_url, plat_ca = sys.argv[1:5]
sys.modules["truststore"] = None
orig_load_default_certs = ssl.SSLContext.load_default_certs
def load_platform_plus_test_root(self, purpose=ssl.Purpose.SERVER_AUTH):
    orig_load_default_certs(self, purpose)
    self.load_verify_locations(cafile=plat_ca)
ssl.SSLContext.load_default_certs = load_platform_plus_test_root
from agent.ssl_verify import resolve_httpx_verify
verify = resolve_httpx_verify()
assert isinstance(verify, ssl.SSLContext), verify
assert httpx.get(env_url, verify=verify, timeout=5).content == b"private provider"
assert httpx.get(plat_url, verify=verify, timeout=5).content == b"private provider"
try:
    httpx.get(untrusted_url, verify=verify, timeout=5)
except httpx.ConnectError:
    pass
else:
    raise AssertionError("untrusted server cert was accepted")
"""
        env = _child_env(tmp_path)
        env["SSL_CERT_FILE"] = str(env_ca)
        child = subprocess.run(
            [
                sys.executable, "-c", script,
                f"https://127.0.0.1:{env_srv.server_port}/",
                f"https://127.0.0.1:{plat_srv.server_port}/",
                f"https://127.0.0.1:{untrusted_srv.server_port}/",
                str(plat_ca),
            ],
            env=env, capture_output=True, text=True, timeout=30, check=False,
        )
        assert child.returncode == 0, child.stdout + child.stderr


def test_resolve_httpx_verify_rejects_untrusted_cert_with_unset_env(tmp_path):
    """With no CA env vars set, resolve_httpx_verify() still rejects an untrusted certificate."""
    import subprocess
    import sys

    with _private_https_server(tmp_path) as (server, _ca):
        script = """
import httpx, sys
from agent.ssl_verify import resolve_httpx_verify
try:
    httpx.get(sys.argv[1], verify=resolve_httpx_verify(), timeout=5)
except httpx.ConnectError:
    pass
else:
    raise AssertionError("untrusted server cert was accepted")
"""
        child = subprocess.run(
            [sys.executable, "-c", script, f"https://127.0.0.1:{server.server_port}/"],
            env=_child_env(tmp_path), capture_output=True, text=True, timeout=30, check=False,
        )
        assert child.returncode == 0, child.stdout + child.stderr


@pytest.mark.parametrize("file_env", ["SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"])
def test_resolve_httpx_verify_unions_file_dir_and_platform_store_and_isolates_cache(
    tmp_path, file_env
):
    """When both a CA file env and SSL_CERT_DIR are set, one union context trusts
    CAs from the file, the dir, and the platform store, rejects untrusted CAs,
    and does not collide in the cache with file-only or dir-only contexts."""
    import shutil
    import subprocess
    import sys

    openssl = shutil.which("openssl")
    if not openssl:
        pytest.skip("openssl CLI required to compute subject hash for SSL_CERT_DIR")

    file_dir = tmp_path / "file_ca"
    dir_ca_src = tmp_path / "dir_ca_src"
    ca_dir = tmp_path / "hashed_certs"
    platform_dir = tmp_path / "platform_ca"
    untrusted_dir = tmp_path / "untrusted_ca"
    for d in (file_dir, dir_ca_src, ca_dir, platform_dir, untrusted_dir):
        d.mkdir()

    with _private_https_server(file_dir, common_name="File CA") as (file_srv, file_ca), \
         _private_https_server(dir_ca_src, common_name="Dir CA") as (dir_srv, dir_ca), \
         _private_https_server(platform_dir, common_name="Platform CA") as (plat_srv, plat_ca), \
         _private_https_server(untrusted_dir, common_name="Untrusted CA") as (untrusted_srv, _untrusted_ca):
        cert_hash = subprocess.check_output(
            [openssl, "x509", "-hash", "-noout", "-in", str(dir_ca)],
            text=True,
        ).strip()
        (ca_dir / f"{cert_hash}.0").symlink_to(dir_ca)
        script = """
import httpx, os, ssl, sys
file_env, file_ca, ca_dir, file_url, dir_url, plat_url, untrusted_url, plat_ca = sys.argv[1:9]
sys.modules["truststore"] = None
orig_load_default_certs = ssl.SSLContext.load_default_certs
def load_platform_plus_test_root(self, purpose=ssl.Purpose.SERVER_AUTH):
    orig_load_default_certs(self, purpose)
    self.load_verify_locations(cafile=plat_ca)
ssl.SSLContext.load_default_certs = load_platform_plus_test_root
from agent.ssl_verify import resolve_httpx_verify

os.environ[file_env] = file_ca
os.environ.pop("SSL_CERT_DIR", None)
file_only_ctx = resolve_httpx_verify()

os.environ.pop(file_env, None)
os.environ["SSL_CERT_DIR"] = ca_dir
dir_only_ctx = resolve_httpx_verify()

os.environ[file_env] = file_ca
os.environ["SSL_CERT_DIR"] = ca_dir
both_ctx = resolve_httpx_verify()

assert both_ctx is not file_only_ctx
assert both_ctx is not dir_only_ctx
assert file_only_ctx is not dir_only_ctx
assert resolve_httpx_verify() is both_ctx

assert httpx.get(file_url, verify=both_ctx, timeout=5).content == b"private provider"
assert httpx.get(dir_url, verify=both_ctx, timeout=5).content == b"private provider"
assert httpx.get(plat_url, verify=both_ctx, timeout=5).content == b"private provider"
try:
    httpx.get(untrusted_url, verify=both_ctx, timeout=5)
except httpx.ConnectError:
    pass
else:
    raise AssertionError("untrusted server cert was accepted")
"""
        env = _child_env(tmp_path)
        child = subprocess.run(
            [
                sys.executable, "-c", script,
                file_env,
                str(file_ca),
                str(ca_dir),
                f"https://127.0.0.1:{file_srv.server_port}/",
                f"https://127.0.0.1:{dir_srv.server_port}/",
                f"https://127.0.0.1:{plat_srv.server_port}/",
                f"https://127.0.0.1:{untrusted_srv.server_port}/",
                str(plat_ca),
            ],
            env=env, capture_output=True, text=True, timeout=30, check=False,
        )
        assert child.returncode == 0, child.stdout + child.stderr
