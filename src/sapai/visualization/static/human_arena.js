(() => {
  "use strict";

  const script = document.currentScript;
  const dataElement = script.previousElementSibling;
  const root = dataElement.previousElementSibling.previousElementSibling;
  const bootstrap = JSON.parse(dataElement.textContent);
  const callbackName = bootstrap.callbackName;
  let view = bootstrap.view;
  let busy = false;
  let selected = null;
  let reorderMode = false;
  let reorderOrder = [];
  let battleIndex = 0;
  let playTimer = null;
  let decisionStarted = performance.now();

  const escapeHtml = (value) => String(value ?? "").replace(
    /[&<>"']/g,
    (character) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    })[character],
  );

  const disabled = () => busy ? " disabled" : "";
  const actionById = (id) => view.actions.find((action) => action.id === id);
  const actionsOf = (kind, source = null) => view.actions.filter(
    (action) => action.kind === kind && (source === null || action.source === source),
  );

  function sprite(kind, name) {
    return view.sprites?.[kind]?.[name] || "";
  }

  function animationClasses(pet, side) {
    const animation = pet?.animation || {};
    return [
      pet?.frozen ? "frozen" : "",
      animation.entered ? "is-summoned" : "",
      animation.healthDelta < 0 ? "is-hurt" : "",
      animation.attackDelta > 0 || animation.healthDelta > 0 ? "is-buffed" : "",
      animation.perkChanged ? "perk-changed" : "",
      animation.role === "attacker" ? "is-attacker" : "",
      animation.role === "target" ? "is-target" : "",
      side ? `side-${side}` : "",
    ].filter(Boolean).join(" ");
  }

  function deltas(pet) {
    const animation = pet?.animation || {};
    const items = [];
    if (animation.healthDelta < 0) {
      items.push(`<span class="delta damage">${animation.healthDelta} ♥</span>`);
    }
    if (animation.attackDelta > 0) {
      items.push(`<span class="delta buff">+${animation.attackDelta} ⚔</span>`);
    }
    if (animation.healthDelta > 0) {
      items.push(`<span class="delta buff">+${animation.healthDelta} ♥</span>`);
    }
    if (animation.perkChanged) {
      const label = pet.perk ? `◆ ${escapeHtml(pet.perk)}` : "Perk used";
      items.push(`<span class="delta perk">${label}</span>`);
    }
    return items.length ? `<div class="delta-stack">${items.join("")}</div>` : "";
  }

  function petCard(pet, options = {}) {
    const classes = [
      "pet",
      options.interactive ? "interactive-card" : "",
      options.front ? "front" : "",
      options.selected ? "selected" : "",
      options.legalTarget ? "legal-target" : "",
      options.orderNumber ? "reorder-picked" : "",
      pet ? animationClasses(pet, options.side) : "empty",
    ].filter(Boolean).join(" ");
    const attributes = [
      options.click ? `data-click="${escapeHtml(options.click)}"` : "",
      options.orderNumber ? `data-order-number="${options.orderNumber}"` : "",
      options.interactive ? 'role="button" tabindex="0"' : "",
    ].filter(Boolean).join(" ");
    if (!pet) {
      return `<div class="${classes}" ${attributes}>Empty</div>`;
    }
    const source = sprite("pet", pet.name);
    const image = source
      ? `<img class="pet-sprite" src="${source}" alt="${escapeHtml(pet.name)}">`
      : '<div style="height:92px"></div>';
    const perk = pet.perk ? `<span class="perk">${escapeHtml(pet.perk)}</span>` : "";
    return `<div class="${classes}" ${attributes} data-pet-id="${pet.visualId ?? ""}">
      <div class="stats"><span class="attack">${Math.max(0, pet.attack)}</span><span class="health">${Math.max(0, pet.health)}</span></div>
      ${deltas(pet)}${image}<div class="name">${escapeHtml(pet.name)}</div>
      <div class="level">Level ${pet.level}</div>${perk}
    </div>`;
  }

  function foodCard(food, options = {}) {
    const source = sprite("food", food.name);
    const image = source
      ? `<img src="${source}" alt="${escapeHtml(food.name)}">`
      : '<div style="height:92px"></div>';
    const classes = [
      "food",
      "interactive-card",
      food.frozen ? "frozen" : "",
      options.selected ? "selected" : "",
    ].filter(Boolean).join(" ");
    return `<div class="${classes}" data-click="food:${options.index}" role="button" tabindex="0">
      ${image}<div class="name">${escapeHtml(food.name)}</div><div class="level">${food.cost} gold</div>
    </div>`;
  }

  function targetActions() {
    if (!selected || reorderMode) {
      return new Map();
    }
    let candidates = [];
    if (selected.type === "shopPet") {
      candidates = view.actions.filter((action) =>
        ["BUY_PET", "BUY_MERGE_PET"].includes(action.kind)
          && action.source === selected.index,
      );
    } else if (selected.type === "food") {
      candidates = actionsOf("BUY_FOOD", selected.index).filter((action) => action.target >= 0);
    } else if (selected.type === "team") {
      candidates = actionsOf("MERGE_BOARD_PET", selected.index);
    }
    return new Map(candidates.map((action) => [action.target, action]));
  }

  function renderTeam(state) {
    const targets = targetActions();
    return state.team.map((pet, position) => {
      const orderIndex = reorderOrder.indexOf(position);
      return petCard(pet, {
        front: position === 0,
        interactive: true,
        click: `team:${position}`,
        selected: selected?.type === "team" && selected.index === position,
        legalTarget: targets.has(position),
        orderNumber: orderIndex >= 0 ? orderIndex + 1 : 0,
      });
    }).join("");
  }

  function actionButton(action, label, style = "") {
    return `<button class="${style}" data-action="${escapeHtml(action.id)}"${disabled()}>${escapeHtml(label)}</button>`;
  }

  function actionName(action, state) {
    const targetPet = action.target >= 0 ? state.team[action.target] : null;
    const target = targetPet ? `${targetPet.name} in slot ${action.target + 1}` : `slot ${action.target + 1}`;
    switch (action.kind) {
      case "BUY_PET": return `Buy into ${target}`;
      case "BUY_MERGE_PET": return `Buy and merge with ${target}`;
      case "BUY_FOOD": return action.target < 0 ? "Buy food" : `Feed ${target}`;
      case "MERGE_BOARD_PET": return `Merge into ${target}`;
      default: return action.kind.replaceAll("_", " ").toLowerCase();
    }
  }

  function contextPanel(state) {
    if (reorderMode) {
      const occupied = state.team.flatMap((pet, index) => pet ? [index] : []);
      const complete = reorderOrder.length === occupied.length;
      const reorder = complete
        ? view.actions.find((action) =>
          action.kind === "REORDER"
            && action.order.length === reorderOrder.length
            && action.order.every((position, index) => position === reorderOrder[index]),
        )
        : null;
      const apply = reorder ? actionButton(reorder, "Apply order") : "";
      const note = complete && !reorder
        ? "That is already the current order. Reset or cancel without spending an action."
        : "Click every occupied team pet in the desired front-to-back order.";
      return `<div class="context-panel"><div class="context-title">Reorder team</div>
        <div class="context-actions">${apply}<button class="secondary" data-command="reset-order"${disabled()}>Reset</button><button class="secondary" data-command="cancel-order"${disabled()}>Cancel</button></div>
        <div class="context-help">${escapeHtml(note)}</div></div>`;
    }
    if (!selected) {
      return `<div class="context-panel"><div class="context-title">Choose a card</div>
        <div class="context-help">Select a shop offer or team pet to see its exact legal actions.</div></div>`;
    }
    let title = "Selected card";
    let candidates = [];
    if (selected.type === "shopPet") {
      const pet = state.shopPets[selected.index];
      title = pet ? `Shop ${pet.name}` : title;
      candidates = view.actions.filter((action) =>
        action.source === selected.index
          && ["BUY_PET", "BUY_MERGE_PET", "FREEZE_PET", "UNFREEZE_PET"].includes(action.kind),
      );
    } else if (selected.type === "food") {
      const food = state.shopFoods[selected.index];
      title = food ? `Shop ${food.name}` : title;
      candidates = view.actions.filter((action) =>
        action.source === selected.index
          && ["BUY_FOOD", "FREEZE_FOOD", "UNFREEZE_FOOD"].includes(action.kind),
      );
    } else {
      const pet = state.team[selected.index];
      title = pet ? `Team ${pet.name}` : `Empty slot ${selected.index + 1}`;
      candidates = view.actions.filter((action) =>
        action.source === selected.index
          && ["SELL_PET", "MERGE_BOARD_PET"].includes(action.kind),
      );
    }
    const controls = candidates.map((action) => {
      const style = action.kind === "SELL_PET" ? "danger" : "";
      return actionButton(action, actionName(action, state), style);
    }).join("");
    return `<div class="context-panel"><div class="context-title">${escapeHtml(title)}</div>
      <div class="context-actions">${controls || "No direct action available"}</div>
      <div class="context-help">Highlighted team cards are valid targets and can be clicked directly.</div></div>`;
  }

  function summaryPanel(summary) {
    return `<div class="benchmark-summary"><div class="section-title">Human benchmark</div>
      <div class="summary-grid">
        <div class="summary-stat"><strong>${summary.games_completed}</strong>completed games</div>
        <div class="summary-stat"><strong>${Number(summary.trophies_mean).toFixed(2)}</strong>mean trophies</div>
        <div class="summary-stat"><strong>${Number(summary.battle_win_rate * 100).toFixed(1)}%</strong>battle win rate</div>
        <div class="summary-stat"><strong>${summary.decisions}</strong>recorded decisions</div>
      </div></div>`;
  }

  function shopView() {
    const state = view.state;
    const pets = state.shopPets.map((pet, index) => petCard(pet, {
      interactive: true,
      click: `shopPet:${index}`,
      selected: selected?.type === "shopPet" && selected.index === index,
    })).join("");
    const foods = state.shopFoods.map((food, index) => foodCard(food, {
      index,
      selected: selected?.type === "food" && selected.index === index,
    })).join("");
    const roll = actionsOf("ROLL")[0];
    const end = actionsOf("END_TURN")[0];
    return `${view.error ? `<div class="human-error">${escapeHtml(view.error)}</div>` : ""}
      <div class="label"><h2>Episode ${view.episode_index + 1} · Turn ${state.turn}</h2><span class="tag">Human shop</span></div>
      <div class="status"><span class="pill">Tier ${state.tier}</span><span class="pill">🪙 ${state.gold}</span><span class="pill">🏆 ${state.trophies}</span><span class="pill">❤️ ${state.lives}</span></div>
      <div class="section-title">Team · front is left</div><div class="team">${renderTeam(state)}</div>
      <div class="section-title">Shop offers</div><div class="shop"><div class="offers">${pets}</div><div class="foods">${foods}</div></div>
      ${contextPanel(state)}
      <div class="human-toolbar">
        ${roll ? actionButton(roll, "↻ Roll (1 gold)") : ""}
        <button class="secondary" data-command="reorder"${disabled()}>Reorder team</button>
        ${end ? actionButton(end, "End turn") : ""}
      </div>
      ${summaryPanel(view.summary)}`;
  }

  function battleTeam(pets, side) {
    const ordered = side === "player" ? [...pets].reverse() : pets;
    const title = side === "player" ? "Player" : "Opponent";
    const cards = ordered.map((pet) => petCard(pet, {
      side,
      front: pet?.position === 0,
    })).join("");
    return `<div class="battle-side battle-side--${side}"><div class="side-name">${title} · front at center</div><div class="battle-team">${cards}</div></div>`;
  }

  function faintGhost(item) {
    const pet = item.pet;
    const source = pet && sprite("pet", pet.name);
    if (!pet || !source) return "";
    return `<div class="faint-ghost side-${item.side}"><img src="${source}" alt=""><span>${escapeHtml(pet.name)} fainted</span></div>`;
  }

  function battleSlide(slide) {
    const recent = slide.log.slice(-8).map((line) => `<div>${escapeHtml(line)}</div>`).join("") || "<div>Battle setup</div>";
    return `<div class="label"><h2>${escapeHtml(slide.label)}</h2><span class="tag">${escapeHtml(slide.event)}</span></div>
      <div class="battlefield">${battleTeam(slide.player, "player")}<div class="versus">VS</div>${battleTeam(slide.opponent, "opponent")}${slide.departed.map(faintGhost).join("")}</div>
      <div class="log">${recent}</div>`;
  }

  function battleView() {
    const battle = view.battle;
    const slides = battle.slides;
    battleIndex = Math.max(0, Math.min(slides.length - 1, battleIndex));
    const outcome = battle.outcome.replaceAll("_", " ");
    return `${view.error ? `<div class="human-error">${escapeHtml(view.error)}</div>` : ""}
      ${battleSlide(slides[battleIndex])}
      <div class="battle-controls">
        <button class="secondary" data-command="battle-prev"${disabled()}>← Back</button>
        <button class="secondary" data-command="battle-play"${disabled()}>${playTimer ? "❚❚ Pause" : "▶ Play"}</button>
        <input class="battle-progress" data-command="battle-range" type="range" min="0" max="${slides.length - 1}" value="${battleIndex}">
        <button class="secondary" data-command="battle-next"${disabled()}>Next →</button>
        <span>${battleIndex + 1} / ${slides.length}</span>
        <button data-command="continue"${disabled()}>Continue · ${escapeHtml(outcome)}</button>
      </div>`;
  }

  function completeView() {
    const state = view.state;
    return `${view.error ? `<div class="human-error">${escapeHtml(view.error)}</div>` : ""}
      <div class="label"><h2>Episode ${view.episode_index + 1} complete</h2><span class="tag">Human benchmark</span></div>
      <div class="status"><span class="pill">🏆 ${state.trophies} trophies</span><span class="pill">❤️ ${state.lives} lives</span><span class="pill">Turn ${state.turn}</span></div>
      <div class="section-title">Final team · front is left</div><div class="team">${renderTeam(state)}</div>
      ${summaryPanel(view.summary)}
      <div class="complete-actions"><button data-command="new-episode"${disabled()}>Start next game</button></div>
      <div class="benchmark-note">This completed game is immutable. Starting the next game advances the open-ended benchmark.</div>`;
  }

  function stopPlayback() {
    if (playTimer) clearTimeout(playTimer);
    playTimer = null;
  }

  function render() {
    const content = view.stage === "shop"
      ? shopView()
      : view.stage === "battle_review"
        ? battleView()
        : completeView();
    root.innerHTML = `<main class="app"><header class="head"><h1>Human Arena benchmark</h1><div class="subtitle">${escapeHtml(view.participant_alias)} · ${escapeHtml(view.pack)} · revision ${view.revision}</div></header><section class="panel">${busy ? '<div class="human-busy">Saving…</div>' : ""}${content}</section></main>`;
    bindEvents();
  }

  function selectCard(kind, index) {
    if (reorderMode && kind === "team") {
      if (!view.state.team[index] || reorderOrder.includes(index)) return;
      reorderOrder.push(index);
      render();
      return;
    }
    const targets = targetActions();
    if (kind === "team" && targets.has(index)) {
      invokeAction(targets.get(index));
      return;
    }
    selected = selected?.type === kind && selected.index === index ? null : {type: kind, index};
    render();
  }

  async function invoke(command, parameters = {}) {
    if (busy) return;
    busy = true;
    stopPlayback();
    render();
    try {
      const response = await google.colab.kernel.invokeFunction(
        callbackName,
        [command, parameters],
        {},
      );
      view = response.data["application/json"];
      selected = null;
      reorderMode = false;
      reorderOrder = [];
      battleIndex = 0;
      decisionStarted = performance.now();
    } catch (error) {
      view.error = `Colab callback failed: ${error}`;
    } finally {
      busy = false;
      render();
    }
  }

  function invokeAction(action) {
    if (!action) return;
    invoke("action", {
      action_id: action.id,
      revision: view.revision,
      elapsed_ms: Math.max(0, performance.now() - decisionStarted),
    });
  }

  function bindEvents() {
    root.querySelectorAll("[data-click]").forEach((element) => {
      const activate = () => {
        const [kind, rawIndex] = element.dataset.click.split(":");
        selectCard(kind, Number(rawIndex));
      };
      element.onclick = activate;
      element.onkeydown = (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          activate();
        }
      };
    });
    root.querySelectorAll("[data-action]").forEach((element) => {
      element.onclick = () => invokeAction(actionById(element.dataset.action));
    });
    root.querySelectorAll("[data-command]").forEach((element) => {
      const command = element.dataset.command;
      if (command === "reorder") element.onclick = () => {
        reorderMode = true;
        selected = null;
        reorderOrder = [];
        render();
      };
      if (command === "reset-order") element.onclick = () => {
        reorderOrder = [];
        render();
      };
      if (command === "cancel-order") element.onclick = () => {
        reorderMode = false;
        reorderOrder = [];
        render();
      };
      if (command === "battle-prev") element.onclick = () => {
        stopPlayback();
        battleIndex = Math.max(0, battleIndex - 1);
        render();
      };
      if (command === "battle-next") element.onclick = () => {
        stopPlayback();
        battleIndex = Math.min(view.battle.slides.length - 1, battleIndex + 1);
        render();
      };
      if (command === "battle-range") element.oninput = (event) => {
        stopPlayback();
        battleIndex = Number(event.target.value);
        render();
      };
      if (command === "battle-play") element.onclick = () => {
        if (playTimer) {
          stopPlayback();
          render();
          return;
        }
        if (battleIndex >= view.battle.slides.length - 1) battleIndex = 0;
        const tick = () => {
          if (battleIndex >= view.battle.slides.length - 1) {
            stopPlayback();
            render();
            return;
          }
          battleIndex += 1;
          render();
          playTimer = setTimeout(tick, 1200);
        };
        playTimer = setTimeout(tick, 1200);
        render();
      };
      if (command === "continue") element.onclick = () => invoke("continue", {revision: view.revision});
      if (command === "new-episode") element.onclick = () => invoke("new_episode", {revision: view.revision});
    });
  }

  render();
})();
