# SPDX-FileCopyrightText: 2025 DSBL-Dev contributors
#
# SPDX-License-Identifier: Apache-2.0

"""
Gate system for semantic processing.
Gate register for automatic discovery.
"""

from gates.security_gate import SecurityGate
from gates.civil_gate import CivilGate

def register_gates(gate_processor):
    """Register all available gates with the processor."""
    gate_processor.register_gate("sec_clean", SecurityGate())
    gate_processor.register_gate("civil", CivilGate())