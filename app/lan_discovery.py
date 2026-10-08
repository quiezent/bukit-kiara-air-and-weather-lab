"""LAN-only DNS-SD for the existing dashboard HTTP service.

No collection, model fitting, requests, or database access happen here. The daemon
refreshes its interfaces after a DHCP/Wi-Fi change independently of HTTP handling.
"""
from __future__ import annotations

import ipaddress
import re
import socket
import threading

SERVICE_TYPE = "_ttdi-weather._tcp.local."
API_PATH = "/api/rlcd/v1"
SCHEMA_VERSION = "1"


def default_hostname() -> str:
    machine = re.sub(r"[^a-z0-9-]", "-", socket.gethostname().lower()).strip("-")
    return "ttdi-" + (machine or "dashboard")[:55] + ".local."


def lan_ipv4_addresses() -> tuple[str, ...]:
    """Use adapter addresses; never publish loopback, APIPA, or stale DNS entries."""
    import ifaddr

    addresses = set()
    for adapter in ifaddr.get_adapters():
        for item in adapter.ips:
            if not isinstance(item.ip, str):
                continue
            address = ipaddress.ip_address(item.ip)
            if (address.version == 4 and not address.is_loopback
                    and not address.is_link_local and not address.is_unspecified
                    and not address.is_multicast):
                addresses.add(str(address))
    return tuple(sorted(addresses))


class LanDiscovery:
    def __init__(self, port: int, *, hostname: str | None = None,
                 service_type: str = SERVICE_TYPE, instance: str | None = None,
                 poll_seconds: float = 30, address_provider=lan_ipv4_addresses):
        if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
            raise ValueError("Discovery requires the bound HTTP server port")
        self.port = port
        self.hostname = hostname or default_hostname()
        if not self.hostname.endswith(".local."):
            raise ValueError("An mDNS hostname must end in .local.")
        self.service_type = service_type
        self.instance = instance or "TTDI Weather " + socket.gethostname()
        self.service_name = self.instance + "." + self.service_type
        self.poll_seconds = max(1, float(poll_seconds))
        self.address_provider = address_provider
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._status = {"available": False, "hostname": self.hostname,
                        "service_type": self.service_type, "service_name": self.service_name,
                        "port": self.port, "api_path": API_PATH,
                        "schema_version": SCHEMA_VERSION, "ipv4": [], "error": None}

    def status(self) -> dict:
        with self._lock:
            return {**self._status, "ipv4": list(self._status["ipv4"])}

    def _set_status(self, available: bool, addresses=(), error=None):
        with self._lock:
            self._status.update(available=available, ipv4=list(addresses), error=error)

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="rlcd-lan-discovery", daemon=True)
            self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self):
        zc = info = None
        registered_addresses = None
        last_error = None

        def close_registration():
            nonlocal zc, info, registered_addresses
            old_zc, old_info = zc, info
            zc = info = registered_addresses = None
            if old_zc is not None:
                try:
                    if old_info is not None:
                        old_zc.unregister_service(old_info)
                finally:
                    old_zc.close()

        try:
            while not self._stop.is_set():
                try:
                    from zeroconf import IPVersion, ServiceInfo, Zeroconf

                    addresses = self.address_provider()
                    if addresses != registered_addresses:
                        # Rebuild sockets as well as records: update_service alone does
                        # not move multicast memberships to a new DHCP address.
                        close_registration()
                        self._set_status(False)
                        if addresses:
                            zc = Zeroconf(interfaces=list(addresses), ip_version=IPVersion.V4Only)
                            info = ServiceInfo(
                                self.service_type, self.service_name, port=self.port,
                                server=self.hostname, parsed_addresses=list(addresses),
                                properties={"api_path": API_PATH, "schema_version": SCHEMA_VERSION},
                                host_ttl=60, other_ttl=60,
                            )
                            zc.register_service(info, allow_name_change=False)
                            registered_addresses = addresses
                            self._set_status(True, addresses)
                            print(f"[RLCD discovery] {self.service_name} -> "
                                  f"{self.hostname}:{self.port} {','.join(addresses)}", flush=True)
                        else:
                            registered_addresses = ()
                            print("[RLCD discovery] No routable LAN IPv4 address; waiting", flush=True)
                    last_error = None
                except Exception as error:
                    try:
                        close_registration()
                    except Exception:
                        pass
                    message = f"{type(error).__name__}: {error}"
                    self._set_status(False, error=message)
                    if message != last_error:
                        print(f"[RLCD discovery] {message}", flush=True)
                    last_error = message
                self._stop.wait(self.poll_seconds)
        finally:
            try:
                close_registration()
            finally:
                self._set_status(False)


def start_discovery(port: int) -> LanDiscovery:
    return LanDiscovery(port).start()
