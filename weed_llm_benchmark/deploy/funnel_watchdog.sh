#!/usr/bin/env bash
# Checks that the dashboard answers from the public internet through Tailscale Funnel, on every ingress
# address public DNS gives for the host, and repairs the node's side when one does not.
#
# 2026-10-29: after the node lost its connection to its home DERP region, Funnel's public ingress
# accepted TCP and closed every TLS handshake; the dashboard stayed reachable inside the tailnet only.
# Turning the funnel off and on and rebinding the node's sockets (tailscale debug rebind / restun) restored
# every ingress address within two minutes. None of this needs root, and none of it takes the node off
# the tailnet (the lab is administered over that tailnet).
#
# Run by funnel-watchdog.timer every 10 minutes; one line per run in ~/funnel_watchdog.log.
set -u
# The body is one compound command, read whole before it runs, so replacing this file while a run
# is in progress cannot make bash execute a fragment of the new file.
{
HOST="${FUNNEL_HOST:-lab-b660m-c.tailfa6424.ts.net}"
TARGET="${FUNNEL_TARGET:-http://127.0.0.1:8000}"
LOG="${FUNNEL_LOG:-$HOME/funnel_watchdog.log}"
TS() { date -u +%FT%TZ; }

has_ipv6() {  # the lab has no IPv6 route; probing AAAA addresses from it would only report false failures
  curl -6 -s -m 8 -o /dev/null https://cloudflare-dns.com/ 2>/dev/null
}

ingress() {   # public A (and, where this machine has IPv6, AAAA) records, not the tailnet's MagicDNS answer
  { dig +short @1.1.1.1 "$HOST" A; dig +short @8.8.8.8 "$HOST" A
    if has_ipv6; then dig +short @1.1.1.1 "$HOST" AAAA; fi; } 2>/dev/null | grep -E '^[0-9a-f.:]+$' | sort -u
}

probe() {     # "<ok> <total> <failed addresses>"
  local ok=0 total=0 bad="" ip code
  for ip in $(ingress); do
    total=$((total + 1))
    if [[ "$ip" == *:* ]]; then code=$(curl -s -m 15 -o /dev/null -w '%{http_code}' --resolve "$HOST:443:[$ip]" "https://$HOST/login")
    else code=$(curl -s -m 15 -o /dev/null -w '%{http_code}' --resolve "$HOST:443:$ip" "https://$HOST/login"); fi
    if [ "$code" = "200" ]; then ok=$((ok + 1)); else bad="$bad $ip=$code"; fi
  done
  echo "$ok $total$bad"
}

read -r ok total bad <<<"$(probe)"
if [ "${total:-0}" -eq 0 ]; then
  echo "$(TS) public DNS gave no address for $HOST (resolver unreachable?); nothing repaired" >> "$LOG"; exit 0
fi
if [ "$ok" -eq "$total" ]; then
  echo "$(TS) ok $ok/$total" >> "$LOG"; exit 0
fi

# 2026-10-30 12:43Z: re-applying the funnel over itself did not bring the ingress back within 90 s;
# turning it off first and then on did, within 90 s (as on 10-29).
echo "$(TS) FAIL $ok/$total:$bad -> funnel off/on, rebind, restun" >> "$LOG"
tailscale funnel --https=443 off >/dev/null 2>&1
sleep 5
tailscale funnel --bg "$TARGET" >/dev/null 2>&1
tailscale debug rebind >/dev/null 2>&1
tailscale debug restun >/dev/null 2>&1
sleep 150
read -r ok total bad <<<"$(probe)"
echo "$(TS) after repair $ok/$total${bad:+:$bad}" >> "$LOG"
[ "$ok" -eq "$total" ]
exit $?
}
