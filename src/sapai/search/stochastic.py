from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Protocol

from sapai.sim.actions import Action, ActionKind
from sapai.sim.models import RunState
from sapai.sim.shop import ShopEnvironment


class Evaluator(Protocol):
    def evaluate(self, state: RunState, actions: list[Action]) -> tuple[list[float], float]:
        """Return action probabilities and a run value in ``[-1, 1]``."""


class UniformEvaluator:
    def evaluate(self, state: RunState, actions: list[Action]) -> tuple[list[float], float]:
        probability = 1.0 / max(1, len(actions))
        return [probability] * len(actions), 0.0


@dataclass(slots=True)
class SearchConfig:
    simulations: int = 32
    candidate_actions: int = 8
    max_depth: int = 15
    c_puct: float = 1.5
    progressive_widening_c: float = 1.0
    progressive_widening_alpha: float = 0.5
    gumbel_scale: float = 1.0


@dataclass(slots=True)
class Node:
    state: RunState
    visits: int = 0
    value_sum: float = 0.0
    expanded: bool = False
    value_prior: float = 0.0
    edges: dict[Action, Edge] = field(default_factory=dict)

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else self.value_prior


@dataclass(slots=True)
class Edge:
    action: Action
    prior: float
    visits: int = 0
    value_sum: float = 0.0
    outcomes: dict[str, Node] = field(default_factory=dict)

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass(frozen=True, slots=True)
class SearchResult:
    action: Action
    visit_counts: dict[Action, int]
    action_values: dict[Action, float]
    root_value: float


class PolicyGuidedSearch:
    """Small-budget MCTS with sampled roll outcomes and progressive widening."""

    def __init__(
        self,
        environment: ShopEnvironment,
        evaluator: Evaluator,
        config: SearchConfig | None = None,
    ) -> None:
        self.environment = environment
        self.evaluator = evaluator
        self.config = config or SearchConfig()
        self.transpositions: dict[str, Node] = {}

    def search(self, state: RunState, *, seed: int | None = None) -> SearchResult:
        rng = random.Random(seed)
        self.transpositions = {}
        root = self._node(state.clone())
        self._expand(root, rng, root=True)
        if not root.edges:
            raise ValueError("cannot search a state with no legal actions")
        for _ in range(self.config.simulations):
            self._simulate(root, rng, seen=set(), depth=0)
        best = max(root.edges.values(), key=lambda edge: (edge.visits, edge.value))
        return SearchResult(
            action=best.action,
            visit_counts={edge.action: edge.visits for edge in root.edges.values()},
            action_values={edge.action: edge.value for edge in root.edges.values()},
            root_value=root.value,
        )

    def _node(self, state: RunState) -> Node:
        key = state.canonical_key()
        node = self.transpositions.get(key)
        if node is None:
            node = Node(state)
            self.transpositions[key] = node
        return node

    def _expand(self, node: Node, rng: random.Random, *, root: bool = False) -> float:
        actions = self.environment.legal_actions(node.state)
        if not actions:
            node.expanded = True
            node.value_prior = self._terminal_value(node.state)
            return node.value_prior
        priors, value = self.evaluator.evaluate(node.state, actions)
        if len(priors) != len(actions):
            raise ValueError("evaluator returned the wrong number of action priors")
        total = sum(max(0.0, value) for value in priors)
        normalized = (
            [max(0.0, prior) / total for prior in priors]
            if total
            else [1.0 / len(actions)] * len(actions)
        )

        scored = []
        for action, prior in zip(actions, normalized, strict=True):
            gumbel = -math.log(-math.log(max(1e-12, rng.random())))
            score = math.log(max(prior, 1e-12))
            if root:
                score += self.config.gumbel_scale * gumbel
            scored.append((score, action, prior))
        scored.sort(key=lambda item: item[0], reverse=True)
        for _, action, prior in scored[: self.config.candidate_actions]:
            node.edges[action] = Edge(action, prior)
        kept_total = sum(edge.prior for edge in node.edges.values())
        for edge in node.edges.values():
            edge.prior /= kept_total
        node.expanded = True
        node.value_prior = float(value)
        return float(value)

    def _simulate(
        self,
        node: Node,
        rng: random.Random,
        *,
        seen: set[str],
        depth: int,
    ) -> float:
        key = node.state.canonical_key()
        if key in seen:
            return node.value
        if depth >= self.config.max_depth or node.state.awaiting_battle:
            value = node.value_prior if node.expanded else self._expand(node, rng)
            node.visits += 1
            node.value_sum += value
            return value
        if not node.expanded:
            value = self._expand(node, rng)
            node.visits += 1
            node.value_sum += value
            return value
        if not node.edges:
            value = self._terminal_value(node.state)
            node.visits += 1
            node.value_sum += value
            return value

        path = seen | {key}
        edge = None
        child = None
        for candidate in sorted(
            node.edges.values(),
            key=lambda item: self._puct(node, item),
            reverse=True,
        ):
            candidate_child = self._outcome(node, candidate, rng)
            if candidate_child.state.canonical_key() in path:
                continue
            edge = candidate
            child = candidate_child
            break
        if edge is None or child is None:
            # Every candidate returns to an ancestor. Treat this branch as a
            # leaf rather than spending the search budget on a reversible loop.
            value = node.value_prior
            node.visits += 1
            node.value_sum += value
            return value
        value = self._simulate(child, rng, seen=path, depth=depth + 1)
        edge.visits += 1
        edge.value_sum += value
        node.visits += 1
        node.value_sum += value
        return value

    def _outcome(self, node: Node, edge: Edge, rng: random.Random) -> Node:
        if edge.action.kind is not ActionKind.ROLL:
            if edge.outcomes:
                return next(iter(edge.outcomes.values()))
            child_state = self.environment.step(node.state, edge.action, rng).state
            child = self._node(child_state)
            edge.outcomes[child_state.canonical_key()] = child
            return child

        allowed = max(
            1,
            int(
                self.config.progressive_widening_c
                * max(1, edge.visits) ** self.config.progressive_widening_alpha
            ),
        )
        if len(edge.outcomes) < allowed:
            # Give every sampled chance outcome an independent RNG stream.
            chance_rng = random.Random(rng.getrandbits(64))
            child_state = self.environment.step(node.state, edge.action, chance_rng).state
            key = child_state.canonical_key()
            child = self._node(child_state)
            edge.outcomes[key] = child
            return child
        return rng.choice(list(edge.outcomes.values()))

    def _puct(self, node: Node, edge: Edge) -> float:
        exploration = (
            self.config.c_puct * edge.prior * math.sqrt(max(1, node.visits)) / (1 + edge.visits)
        )
        return edge.value + exploration

    @staticmethod
    def _terminal_value(state: RunState) -> float:
        if state.trophies >= 10:
            return 1.0
        if state.lives <= 0:
            return -1.0
        return 0.0
