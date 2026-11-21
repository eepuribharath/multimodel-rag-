# Network Operations Runbook

## Rebooting router RTR-77

To restart edge router RTR-77: drain traffic by lowering its BGP local-pref
to 50, wait for sessions to move (about 3 minutes), then power-cycle via the
management card. Never hard-reboot RTR-77 while it holds the primary uplink.

## VLAN assignments

Guest wireless traffic is isolated on VLAN 210. Corporate laptops use
VLAN 120, and VoIP phones use VLAN 130 with DSCP EF marking.

## Incident severities

A total loss of the primary uplink is a SEV2 unless a customer-facing
service is degraded, which escalates it to SEV1.
