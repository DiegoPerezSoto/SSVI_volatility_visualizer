"""Reference-counted market-data subscription manager for IBKR.

Multiple components (underlying tracker, volatility radar, portfolio valuator)
may need live streams on the same contract. This module de-duplicates requests:
each contract is subscribed to exactly once, only cancelled when every owner
releases it. This keeps the bot well under IBKR's simultaneous market-data
line limit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Set

from ib_insync import IB, Contract, Ticker

logger = logging.getLogger(__name__)


@dataclass
class Subscription:
    """Pairs a live ticker with the set of owners currently relying on it."""

    ticker: Ticker
    owners: Set[str] = field(default_factory=set)


class SubscriptionManager:
    """De-duplicates and reference-counts IBKR market-data subscriptions."""

    def __init__(self, ib: IB) -> None:
        self._ib = ib
        self._subscriptions: Dict[int, Subscription] = {}

    def subscribe(self, contract: Contract, owner: str) -> Ticker:
        """Subscribes to market data on behalf of an owner.

        If already streaming, the owner is added to the existing subscription's
        owner set and no redundant request is sent to IBKR.

        Args:
            contract: Qualified IBKR contract to subscribe to.
            owner: Consumer identifier (e.g. 'radar', 'portfolio', 'underlying').

        Returns:
            The live ib_insync Ticker for this contract.
        """
        con_id = contract.conId
        if con_id in self._subscriptions:
            self._subscriptions[con_id].owners.add(owner)
            return self._subscriptions[con_id].ticker

        ticker = self._ib.reqMktData(contract, "", False, False)
        self._subscriptions[con_id] = Subscription(ticker=ticker, owners={owner})
        return ticker

    def unsubscribe(self, contract: Contract) -> None:
        """Force-cancels a subscription, regardless of remaining owners."""
        con_id = contract.conId
        if con_id in self._subscriptions:
            self._ib.cancelMktData(contract)
            del self._subscriptions[con_id]

    def release_owner(self, owner: str) -> None:
        """Releases every subscription held by an owner.

        Subscriptions reaching zero owners are cancelled with IBKR and removed.

        Args:
            owner: Consumer identifier releasing its subscriptions.
        """
        con_ids_to_remove = []
        for con_id, sub in self._subscriptions.items():
            if owner in sub.owners:
                sub.owners.discard(owner)
                if not sub.owners:
                    con_ids_to_remove.append(con_id)

        for con_id in con_ids_to_remove:
            sub = self._subscriptions[con_id]
            self._ib.cancelMktData(sub.ticker.contract)
            del self._subscriptions[con_id]

    def get_ticker(self, contract: Contract) -> Optional[Ticker]:
        """Returns the live ticker for a contract, or None if not subscribed."""
        sub = self._subscriptions.get(contract.conId)
        return sub.ticker if sub else None

    @property
    def subscriptions(self) -> Dict[int, Subscription]:
        """Exposes the subscription table for callers that need to reconcile holdings."""
        return self._subscriptions

    @property
    def total_subscriptions(self) -> int:
        """Number of distinct contracts currently streaming."""
        return len(self._subscriptions)
