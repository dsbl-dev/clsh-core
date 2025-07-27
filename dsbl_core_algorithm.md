# DSBL Core Algorithm

```pseudocode
# SPDX-License-Identifier: Apache-2.0
# Copyright © 2025 DSBL-Dev contributors
#
# DSBL — Deferred Semantic Binding Language · Core Algorithm (v1.0.1)
# DOI   : 10.5281/zenodo.15742504      ← concept DOI (always latest)
#
# ---------------------------------------------------------------------
# OVERVIEW
# ---------------------------------------------------------------------
# DSBL is a *general-purpose algorithm* for deferred semantic binding:
#   • Symbol semantics bind only at **runtime context** — never at design time.
#   • The algorithm is domain-agnostic; any system requiring context-dependent
#     meaning (multi-agent chat, IoT policy, quantum control, etc.) can adopt it.
#
# The repository you are reading applies DSBL to a *multi-agent social immune
# system* — merely one illustrative use-case, not a limitation.
#
# Two nested fractal logics co-exist:
#   1. 4-Fractal         : deferred binding across temporal scales
#   2. Core/Periphery    : deterministic core  ↔ stochastic periphery
#
# This pseudocode specifies layer 2 (Core / Periphery).  Layer 1 emerges
# implicitly through the symbol flow HANDLE_MESSAGE → GATE_DECIDE → LOG.
#
# ---------------------------------------------------------------------
# CORE PRINCIPLE  (Core/Periphery Fractal)
# ---------------------------------------------------------------------
#   Core (C) :  GATE_DECIDE  +  IMMUNE_ADJUST     ← provably deterministic
#   Periph(P):  BIND_WEIGHT  +  symbol variation  ← adaptive / stochastic
#
# Each HANDLE_MESSAGE call yields a micro-fractal:
#        [  C  →  P ]   ×   4-Fractal sequence   ⇒   system-level dynamics
#
# ---------------------------------------------------------------------
# PSEUDOCODE
# ---------------------------------------------------------------------
procedure HANDLE_MESSAGE(author, raw_text, t_now):
    #: Extract symbols embedded in the raw text
    syms ← PARSE_SYMBOLS(raw_text)

    #: Build runtime context (role, reputation, global pressure, timestamp)
    ctx  ← CURRENT_CONTEXT(author, t_now)

    #: Deferred binding loop
    for s in syms:

        # ---------- Core layer (deterministic) ----------
        if s.type == "GATE":
            if not GATE_DECIDE(s, ctx):
                reject_message()
                return

        # ---------- Periphery layer (adaptive) ----------
        else if s.type == "VOTE":
            weight ← BIND_WEIGHT(s, ctx)
            APPLY_VOTE(author, s.target, weight)

    #: Post-message coordination (core)
    IMMUNE_ADJUST(ctx)

    #: Audit trail (hash + timestamp)
    LOG_EVENT(author, raw_text, ctx)
    return ACCEPTED


# -------------------------------------------------------
# GATE_DECIDE — deterministic pointer-state
# -------------------------------------------------------
procedure GATE_DECIDE(symbol, ctx):
    switch symbol.subtype:
        case "CIVIL": return TOXICITY_SCORE(ctx.text) < cfg.tox_thr
        case "SEC"  : return SAFE_REGEX(ctx.text)
        default     : return true         # defensive default-allow


# -------------------------------------------------------
# IMMUNE_ADJUST — multi-agent frequency controller
# -------------------------------------------------------
procedure IMMUNE_ADJUST(ctx):
    pressure ← PROMO_PRESSURE_LAST_N(ctx.window)   # 12-ticket window by default
    if pressure > HIGH:
        BOOST_FREQUENCIES(adaptive_agents, +Δ)
    else if pressure < LOW:
        REDUCE_FREQUENCIES(adaptive_agents, −Δ)


# -------------------------------------------------------
# BIND_WEIGHT — runtime semantic binding (periphery)
# -------------------------------------------------------
procedure BIND_WEIGHT(vote_symbol, ctx):
    base ← 1
    if ctx.author_status == "BINDER":
        base ← base * 1.5            # privilege amplification
    if ctx.target_reputation < cfg.min_rep:
        adj  ← (ctx.target_reputation - cfg.min_rep) * cfg.scale
        base ← max(0.1, base + adj)
    return base
```