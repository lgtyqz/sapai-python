(() => {
  "use strict";

  const DATA = JSON.parse(document.querySelector("#sapai-data").textContent);
  const requestedFrame = Number(new URLSearchParams(location.search).get("frame") || 0);
  let index = Math.max(0, Math.min(DATA.slides.length - 1, requestedFrame));
  let timer = null;
  const stage = document.querySelector("#stage");
  const range = document.querySelector("#range");
  const play = document.querySelector("#play");
  const speed = document.querySelector("#speed");

  document.querySelector("#title").textContent = DATA.title;
  document.querySelector("#subtitle").textContent = DATA.subtitle;
  range.max = Math.max(0, DATA.slides.length - 1);

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

  function experienceLabel(pet) {
    const experience = Math.max(0, Number(pet.experience) || 0);
    if (experience >= 5) return `Level 3 · XP ${experience}/5 (max)`;
    const nextLevel = experience >= 2 ? 5 : 2;
    return `Level ${pet.level} · XP ${experience}/${nextLevel}`;
  }

  function petCard(pet, options = {}) {
    if (!pet) {
      return '<div class="pet empty">Empty</div>';
    }
    const source = DATA.sprites.pet[pet.name];
    const front = options.front ?? pet.position === 0;
    const image = source
      ? `<img class="pet-sprite" src="${source}" alt="${escapeHtml(pet.name)}">`
      : '<div style="height:92px"></div>';
    const perk = pet.perk ? `<span class="perk">${escapeHtml(pet.perk)}</span>` : "";
    return `<div class="pet ${front ? "front" : ""} ${animationClasses(pet, options.side)}" data-pet-id="${pet.visualId ?? ""}">
      <div class="stats"><span class="attack">${Math.max(0, pet.attack)}</span><span class="health">${Math.max(0, pet.health)}</span></div>
      ${deltas(pet)}${image}<div class="name">${escapeHtml(pet.name)}</div>
      <div class="level" title="Total experience">${experienceLabel(pet)}</div>${perk}
    </div>`;
  }

  function foodCard(food) {
    const source = DATA.sprites.food[food.name];
    const image = source
      ? `<img src="${source}" alt="${escapeHtml(food.name)}">`
      : '<div style="height:92px"></div>';
    return `<div class="food ${food.frozen ? "frozen" : ""}">${image}<div class="name">${escapeHtml(food.name)}</div><div class="level">${food.cost} gold</div></div>`;
  }

  function shopTeam(pets, title) {
    const cards = pets.map(
      (pet, position) => petCard(pet, {front: position === 0}),
    ).reverse().join("");
    return `<div class="section-title">${title} · front is right</div><div class="team">${cards}</div>`;
  }

  function battleTeam(pets, side) {
    const ordered = side === "player" ? [...pets].reverse() : pets;
    const title = side === "player" ? "Player" : "Opponent";
    const cards = ordered.map((pet) => petCard(
      pet,
      {side, front: pet?.position === 0},
    )).join("");
    return `<div class="battle-side battle-side--${side}"><div class="side-name">${title} · front at center</div><div class="battle-team">${cards}</div></div>`;
  }

  function faintGhost(item) {
    const pet = item.pet;
    const source = pet && DATA.sprites.pet[pet.name];
    if (!pet || !source) {
      return "";
    }
    return `<div class="faint-ghost side-${item.side}"><img src="${source}" alt=""><span>${escapeHtml(pet.name)} fainted</span></div>`;
  }

  function battle(slide) {
    const recent = slide.log.slice(-8).map(
      (line) => `<div>${escapeHtml(line)}</div>`,
    ).join("") || "<div>Battle setup</div>";
    return `<div class="label"><h2>${escapeHtml(slide.label)}</h2><span class="tag">${escapeHtml(slide.event)}</span></div>
      <div class="battlefield">${battleTeam(slide.player, "player")}<div class="versus">VS</div>${battleTeam(slide.opponent, "opponent")}${slide.departed.map(faintGhost).join("")}</div>
      <div class="log">${recent}</div>`;
  }

  function shop(slide) {
    const state = slide.state;
    const action = slide.action ? `<div class="action">${escapeHtml(slide.action)}</div>` : "";
    const pets = state.shopPets.map((pet) => petCard(pet)).join("");
    const foods = state.shopFoods.map(foodCard).join("");
    return `<div class="label"><h2>${escapeHtml(slide.label)}</h2><span class="tag">Shop</span></div>${action}
      <div class="status"><span class="pill">Turn ${state.turn}</span><span class="pill">Tier ${state.tier}</span><span class="pill">🪙 ${state.gold}</span><span class="pill">🏆 ${state.trophies}</span><span class="pill">❤️ ${state.lives}</span></div>
      ${shopTeam(state.team, "Team")}<div class="section-title">Shop offers</div>
      <div class="shop"><div class="offers">${pets}</div><div class="foods">${foods}</div></div>`;
  }

  function render() {
    if (!DATA.slides.length) {
      stage.innerHTML = "<h2>No timeline frames</h2>";
      return;
    }
    const slide = DATA.slides[index];
    stage.innerHTML = slide.type === "battle" ? battle(slide) : shop(slide);
    range.value = index;
    document.querySelector("#counter").textContent = `${index + 1} / ${DATA.slides.length}`;
    document.querySelector("#prev").disabled = index === 0;
    document.querySelector("#next").disabled = index === DATA.slides.length - 1;
  }

  function stop() {
    if (timer) {
      clearTimeout(timer);
    }
    timer = null;
    play.textContent = "▶ Play";
  }

  function tick() {
    if (index >= DATA.slides.length - 1) {
      stop();
      return;
    }
    index += 1;
    render();
    timer = setTimeout(tick, Number(speed.value));
  }

  function start() {
    if (index >= DATA.slides.length - 1) {
      index = 0;
    }
    play.textContent = "❚❚ Pause";
    render();
    timer = setTimeout(tick, Number(speed.value));
  }

  document.querySelector("#prev").onclick = () => {
    stop();
    index = Math.max(0, index - 1);
    render();
  };
  document.querySelector("#next").onclick = () => {
    stop();
    index = Math.min(DATA.slides.length - 1, index + 1);
    render();
  };
  play.onclick = () => timer ? stop() : start();
  speed.onchange = () => {
    if (timer) {
      stop();
      start();
    }
  };
  range.oninput = (event) => {
    stop();
    index = Number(event.target.value);
    render();
  };
  document.addEventListener("keydown", (event) => {
    if (event.key === "ArrowLeft") {
      document.querySelector("#prev").click();
    }
    if (event.key === "ArrowRight") {
      document.querySelector("#next").click();
    }
    if (event.key === " ") {
      event.preventDefault();
      play.click();
    }
  });
  render();
})();
