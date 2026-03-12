"""Unit tests for src.execution.lifecycle_transitions — order state machine.

Covers:
- DB status constants
- ALLOWED_DB_TRANSITIONS mapping completeness
- is_allowed_transition(): all valid transitions, all invalid transitions
- is_allowed_transition_domain(): OrderStatus enum to DB mapping
- Terminal states (FILLED, REJECTED, CANCELLED) have no outgoing transitions
- ORDER_STATUS_TO_DB mapping completeness
"""

import pytest

from src.core.events import OrderStatus
from src.execution.lifecycle_transitions import (
    ALLOWED_DB_TRANSITIONS,
    DB_ACK,
    DB_CANCELLED,
    DB_FILLED,
    DB_NEW,
    DB_PARTIAL,
    DB_REJECTED,
    DB_SUBMITTING,
    ORDER_STATUS_TO_DB,
    is_allowed_transition,
    is_allowed_transition_domain,
)

# ──────────────────────────────────────────────────
# DB status constants
# ──────────────────────────────────────────────────


class TestDBStatusConstants:
    def test_status_values(self):
        assert DB_SUBMITTING == "SUBMITTING"
        assert DB_NEW == "NEW"
        assert DB_ACK == "ACK"
        assert DB_PARTIAL == "PARTIAL"
        assert DB_FILLED == "FILLED"
        assert DB_REJECTED == "REJECTED"
        assert DB_CANCELLED == "CANCELLED"


# ──────────────────────────────────────────────────
# Transition map completeness
# ──────────────────────────────────────────────────


class TestTransitionMap:
    def test_all_statuses_have_entries(self):
        all_statuses = {DB_SUBMITTING, DB_NEW, DB_ACK, DB_PARTIAL, DB_FILLED, DB_REJECTED, DB_CANCELLED}
        assert set(ALLOWED_DB_TRANSITIONS.keys()) == all_statuses

    def test_terminal_states_have_no_transitions(self):
        assert ALLOWED_DB_TRANSITIONS[DB_FILLED] == set()
        assert ALLOWED_DB_TRANSITIONS[DB_REJECTED] == set()
        assert ALLOWED_DB_TRANSITIONS[DB_CANCELLED] == set()


# ──────────────────────────────────────────────────
# is_allowed_transition (DB strings)
# ──────────────────────────────────────────────────


class TestIsAllowedTransition:
    # --- SUBMITTING transitions ---
    @pytest.mark.parametrize(
        "to_status",
        [DB_NEW, DB_ACK, DB_REJECTED, DB_CANCELLED],
    )
    def test_submitting_valid_transitions(self, to_status):
        assert is_allowed_transition(DB_SUBMITTING, to_status) is True

    def test_submitting_cannot_go_to_filled(self):
        assert is_allowed_transition(DB_SUBMITTING, DB_FILLED) is False

    def test_submitting_cannot_go_to_partial(self):
        assert is_allowed_transition(DB_SUBMITTING, DB_PARTIAL) is False

    # --- NEW transitions ---
    @pytest.mark.parametrize(
        "to_status",
        [DB_ACK, DB_PARTIAL, DB_FILLED, DB_REJECTED, DB_CANCELLED],
    )
    def test_new_valid_transitions(self, to_status):
        assert is_allowed_transition(DB_NEW, to_status) is True

    def test_new_cannot_go_to_submitting(self):
        assert is_allowed_transition(DB_NEW, DB_SUBMITTING) is False

    # --- ACK transitions ---
    @pytest.mark.parametrize(
        "to_status",
        [DB_PARTIAL, DB_FILLED, DB_REJECTED, DB_CANCELLED],
    )
    def test_ack_valid_transitions(self, to_status):
        assert is_allowed_transition(DB_ACK, to_status) is True

    def test_ack_cannot_go_back_to_new(self):
        assert is_allowed_transition(DB_ACK, DB_NEW) is False

    # --- PARTIAL transitions ---
    def test_partial_can_fill(self):
        assert is_allowed_transition(DB_PARTIAL, DB_FILLED) is True

    def test_partial_can_cancel(self):
        assert is_allowed_transition(DB_PARTIAL, DB_CANCELLED) is True

    def test_partial_cannot_reject(self):
        assert is_allowed_transition(DB_PARTIAL, DB_REJECTED) is False

    def test_partial_cannot_go_to_new(self):
        assert is_allowed_transition(DB_PARTIAL, DB_NEW) is False

    # --- Terminal states ---
    @pytest.mark.parametrize("terminal", [DB_FILLED, DB_REJECTED, DB_CANCELLED])
    @pytest.mark.parametrize(
        "to_status",
        [DB_SUBMITTING, DB_NEW, DB_ACK, DB_PARTIAL, DB_FILLED, DB_REJECTED, DB_CANCELLED],
    )
    def test_terminal_cannot_transition(self, terminal, to_status):
        assert is_allowed_transition(terminal, to_status) is False

    # --- Unknown status ---
    def test_unknown_from_status_returns_false(self):
        assert is_allowed_transition("UNKNOWN", DB_NEW) is False

    def test_unknown_to_status_returns_false(self):
        assert is_allowed_transition(DB_NEW, "UNKNOWN") is False


# ──────────────────────────────────────────────────
# is_allowed_transition_domain (OrderStatus enums)
# ──────────────────────────────────────────────────


class TestIsAllowedTransitionDomain:
    def test_submitting_to_live(self):
        assert is_allowed_transition_domain(OrderStatus.SUBMITTING, OrderStatus.LIVE) is True

    def test_submitting_to_rejected(self):
        assert is_allowed_transition_domain(OrderStatus.SUBMITTING, OrderStatus.REJECTED) is True

    def test_live_to_filled(self):
        assert is_allowed_transition_domain(OrderStatus.LIVE, OrderStatus.FILLED) is True

    def test_live_to_partially_filled(self):
        assert is_allowed_transition_domain(OrderStatus.LIVE, OrderStatus.PARTIALLY_FILLED) is True

    def test_partially_filled_to_filled(self):
        assert is_allowed_transition_domain(OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED) is True

    def test_filled_cannot_cancel(self):
        assert is_allowed_transition_domain(OrderStatus.FILLED, OrderStatus.CANCELLED) is False

    def test_cancelled_cannot_fill(self):
        assert is_allowed_transition_domain(OrderStatus.CANCELLED, OrderStatus.FILLED) is False

    def test_pending_to_live(self):
        # PENDING maps to DB_NEW, LIVE maps to DB_ACK; NEW -> ACK is valid
        assert is_allowed_transition_domain(OrderStatus.PENDING, OrderStatus.LIVE) is True

    def test_timeout_uncertain_is_terminal(self):
        # TIMEOUT_UNCERTAIN maps to DB_CANCELLED (terminal)
        assert is_allowed_transition_domain(OrderStatus.TIMEOUT_UNCERTAIN, OrderStatus.FILLED) is False

    def test_expired_is_terminal(self):
        # EXPIRED maps to DB_CANCELLED (terminal)
        assert is_allowed_transition_domain(OrderStatus.EXPIRED, OrderStatus.LIVE) is False


# ──────────────────────────────────────────────────
# ORDER_STATUS_TO_DB mapping
# ──────────────────────────────────────────────────


class TestOrderStatusToDBMapping:
    def test_all_order_statuses_mapped(self):
        for status in OrderStatus:
            assert status in ORDER_STATUS_TO_DB, f"{status} not mapped to DB status"

    def test_pending_maps_to_new(self):
        assert ORDER_STATUS_TO_DB[OrderStatus.PENDING] == DB_NEW

    def test_submitting_maps_to_submitting(self):
        assert ORDER_STATUS_TO_DB[OrderStatus.SUBMITTING] == DB_SUBMITTING

    def test_live_maps_to_ack(self):
        assert ORDER_STATUS_TO_DB[OrderStatus.LIVE] == DB_ACK

    def test_filled_maps_to_filled(self):
        assert ORDER_STATUS_TO_DB[OrderStatus.FILLED] == DB_FILLED
