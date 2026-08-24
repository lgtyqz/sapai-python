"""Render battle and Arena timelines as self-contained HTML files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sapai.sim.actions import Action
from sapai.sim.battle import BattleFrame, BattleResult
from sapai.sim.models import Food, Pet, RunState, Team
from sapai.training.arena import ArenaRunResult
from sapai.visualization.assets import SpriteAtlas


def _pet(pet: Pet | None) -> dict[str, Any] | None:
    if pet is None:
        return None
    return {
        "name": pet.name,
        "attack": pet.effective_attack,
        "health": pet.effective_health,
        "level": pet.level,
        "experience": pet.experience,
        "perk": pet.perk,
    }


def _team(team: Team) -> list[dict[str, Any] | None]:
    return [_pet(pet) for pet in team.slots]


def _food(food: Food) -> dict[str, Any]:
    return {"name": food.name, "cost": food.cost, "frozen": food.frozen}


def _state(state: RunState) -> dict[str, Any]:
    return {
        "turn": state.turn,
        "gold": state.gold,
        "lives": state.lives,
        "trophies": state.trophies,
        "tier": state.tier,
        "team": _team(state.team),
        "shopPets": [
            {**_pet(offer.pet), "frozen": offer.frozen}  # type: ignore[arg-type]
            for offer in state.shop.pets
        ],
        "shopFoods": [_food(food) for food in state.shop.foods],
    }


def _battle_slide(frame: BattleFrame, result: BattleResult, *, prefix: str = ""):
    return {
        "type": "battle",
        "label": f"{prefix}{frame.label}",
        "player": _team(frame.player),
        "opponent": _team(frame.opponent),
        "log": result.log[: frame.log_index],
        "outcome": result.outcome.value,
    }


def _action(action: Action | None) -> str | None:
    if action is None:
        return None
    details = []
    if action.source >= 0:
        details.append(f"source {action.source + 1}")
    if action.target >= 0:
        details.append(f"target {action.target + 1}")
    if action.order:
        details.append("order " + "→".join(str(value + 1) for value in action.order))
    suffix = f" ({', '.join(details)})" if details else ""
    return action.kind.name.replace("_", " ").title() + suffix


def render_battle_html(
    result: BattleResult,
    output_path: str | Path,
    assets_root: str | Path,
) -> Path:
    frames = result.frames or [
        BattleFrame("Battle result", result.player, result.opponent, len(result.log))
    ]
    slides = [_battle_slide(frame, result) for frame in frames]
    payload = {
        "title": "Super Auto Pets battle",
        "subtitle": f"{result.outcome.value.replace('_', ' ').title()} · {result.rounds} rounds",
        "slides": slides,
    }
    return _write_html(payload, output_path, assets_root)


def render_arena_html(
    run: ArenaRunResult,
    output_path: str | Path,
    assets_root: str | Path,
) -> Path:
    slides: list[dict[str, Any]] = []
    for arena_turn in run.turns:
        for frame in arena_turn.shop_frames:
            slides.append(
                {
                    "type": "shop",
                    "label": frame.label,
                    "action": _action(frame.action),
                    "state": _state(frame.state),
                }
            )
        if arena_turn.battle:
            for frame in arena_turn.battle.frames:
                slides.append(
                    _battle_slide(
                        frame,
                        arena_turn.battle,
                        prefix=f"Turn {arena_turn.turn} · ",
                    )
                )
    final = run.final_state
    payload = {
        "title": "Super Auto Pets Arena run",
        "subtitle": (
            f"{final.trophies} trophies · {final.lives} lives · "
            f"{len(run.turns)} battles"
        ),
        "slides": slides,
    }
    return _write_html(payload, output_path, assets_root)


def _collect_names(slides: list[dict[str, Any]]) -> dict[str, set[str]]:
    names = {"pet": set(), "food": set(), "toy": set()}
    for slide in slides:
        if slide["type"] == "battle":
            pets = slide["player"] + slide["opponent"]
            names["pet"].update(pet["name"] for pet in pets if pet)
        else:
            state = slide["state"]
            names["pet"].update(pet["name"] for pet in state["team"] if pet)
            names["pet"].update(pet["name"] for pet in state["shopPets"])
            names["food"].update(food["name"] for food in state["shopFoods"])
    return names


def _write_html(
    payload: dict[str, Any],
    output_path: str | Path,
    assets_root: str | Path,
) -> Path:
    atlas = SpriteAtlas(assets_root)
    payload["sprites"] = atlas.payload(_collect_names(payload["slides"]))
    encoded = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_TEMPLATE.replace("__SAPAI_PAYLOAD__", encoded), encoding="utf-8")
    return destination


_TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SAP AI visualization</title>
<style>
:root{--ink:#20302d;--muted:#6f7e79;--paper:#fbf7eb;--card:#fffdf7;--green:#4f8f79;--gold:#e8b44e;--red:#d8645b}
*{box-sizing:border-box}body{margin:0;color:var(--ink);font-family:ui-rounded,"SF Pro Rounded",system-ui,sans-serif;background:radial-gradient(circle at top,#fff7ce,#b9ded0 62%,#6ca58f);min-height:100vh}
.app{max-width:1180px;margin:auto;padding:24px}.head,.controls,.panel{background:rgba(255,253,247,.94);border:2px solid rgba(32,48,45,.15);box-shadow:0 14px 36px rgba(31,73,61,.16);border-radius:24px}.head{padding:20px 26px;margin-bottom:14px}.head h1{margin:0;font-size:clamp(25px,4vw,42px)}.subtitle{color:var(--muted);margin-top:5px}.controls{display:flex;align-items:center;gap:12px;padding:10px 14px;margin-bottom:14px}.controls button{border:0;border-radius:14px;background:var(--green);color:white;padding:10px 18px;font-weight:800;cursor:pointer}.controls button:disabled{opacity:.35}.controls input{flex:1}.counter{min-width:74px;text-align:right;font-variant-numeric:tabular-nums}.panel{padding:20px;min-height:530px}.label{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:14px}.label h2{margin:0;font-size:25px}.tag{background:#e9f3ee;border-radius:999px;padding:7px 12px;font-size:13px;font-weight:800;text-transform:uppercase;letter-spacing:.06em}.status{display:flex;gap:9px;flex-wrap:wrap;margin-bottom:17px}.pill{background:#f4ecd3;border-radius:999px;padding:7px 12px;font-weight:750}.section-title{margin:16px 0 8px;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.1em}.team{display:grid;grid-template-columns:repeat(5,minmax(100px,1fr));gap:10px}.pet,.food{position:relative;min-height:154px;padding:8px;border:2px solid #d9d5c7;background:var(--card);border-radius:18px;text-align:center;overflow:hidden}.pet.front{border-color:var(--gold)}.pet img,.food img{width:92px;height:92px;object-fit:contain;filter:drop-shadow(0 5px 3px #0002)}.pet .name,.food .name{font-size:13px;font-weight:800;line-height:1.1}.stats{position:absolute;left:7px;right:7px;top:7px;display:flex;justify-content:space-between;font-size:14px;font-weight:900}.attack{color:white;background:#d36b38;border-radius:999px;padding:3px 7px}.health{color:white;background:#5a9b67;border-radius:999px;padding:3px 7px}.level{font-size:11px;color:var(--muted);margin-top:4px}.perk{position:absolute;right:5px;bottom:5px;background:#725b9d;color:white;border-radius:8px;padding:2px 5px;font-size:10px}.empty{display:grid;place-items:center;color:#aaa;background:rgba(255,255,255,.36);border-style:dashed}.shop{display:grid;grid-template-columns:2fr 1fr;gap:14px}.offers{display:grid;grid-template-columns:repeat(5,minmax(90px,1fr));gap:8px}.foods{display:grid;grid-template-columns:repeat(2,minmax(90px,1fr));gap:8px}.food{min-height:144px}.frozen:after{content:"❄";position:absolute;top:6px;right:7px;font-size:22px}.versus{font-weight:900;text-align:center;font-size:20px;margin:12px}.log{margin-top:14px;background:#263b36;color:#eaf6f1;padding:13px 16px;border-radius:16px;max-height:145px;overflow:auto;font:12px/1.5 ui-monospace,monospace}.log div:last-child{color:#ffd778}.action{margin:-5px 0 13px;color:var(--green);font-weight:850}
@media(max-width:760px){.app{padding:10px}.panel{padding:12px}.team,.offers{grid-template-columns:repeat(3,1fr)}.shop{grid-template-columns:1fr}.pet{min-height:140px}.pet img,.food img{width:76px;height:76px}}
</style></head><body><main class="app"><header class="head"><h1 id="title"></h1><div class="subtitle" id="subtitle"></div></header><nav class="controls"><button id="prev">← Back</button><input id="range" type="range" min="0" value="0"><button id="next">Next →</button><span class="counter" id="counter"></span></nav><section class="panel" id="stage"></section></main>
<script>const DATA=__SAPAI_PAYLOAD__;let index=0;const stage=document.querySelector('#stage'),range=document.querySelector('#range');document.querySelector('#title').textContent=DATA.title;document.querySelector('#subtitle').textContent=DATA.subtitle;range.max=Math.max(0,DATA.slides.length-1);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function petCard(p,i){if(!p)return '<div class="pet empty">Empty</div>';const src=DATA.sprites.pet[p.name];return `<div class="pet ${i===0?'front':''} ${p.frozen?'frozen':''}"><div class="stats"><span class="attack">${p.attack}</span><span class="health">${p.health}</span></div>${src?`<img src="${src}" alt="">`:'<div style="height:92px"></div>'}<div class="name">${esc(p.name)}</div><div class="level">Level ${p.level}</div>${p.perk?`<span class="perk">${esc(p.perk)}</span>`:''}</div>`}
function foodCard(f){const src=DATA.sprites.food[f.name];return `<div class="food ${f.frozen?'frozen':''}">${src?`<img src="${src}" alt="">`:'<div style="height:92px"></div>'}<div class="name">${esc(f.name)}</div><div class="level">${f.cost} gold</div></div>`}
const team=(pets,title)=>`<div class="section-title">${title} · front is left</div><div class="team">${pets.map(petCard).join('')}</div>`;
function battle(s){const recent=s.log.slice(-8).map(x=>`<div>${esc(x)}</div>`).join('')||'<div>Battle setup</div>';return `<div class="label"><h2>${esc(s.label)}</h2><span class="tag">${esc(s.outcome.replaceAll('_',' '))}</span></div>${team(s.opponent,'Opponent')}<div class="versus">VS</div>${team(s.player,'Player')}<div class="log">${recent}</div>`}
function shop(s){const x=s.state;return `<div class="label"><h2>${esc(s.label)}</h2><span class="tag">Shop</span></div>${s.action?`<div class="action">${esc(s.action)}</div>`:''}<div class="status"><span class="pill">Turn ${x.turn}</span><span class="pill">Tier ${x.tier}</span><span class="pill">🪙 ${x.gold}</span><span class="pill">🏆 ${x.trophies}</span><span class="pill">❤️ ${x.lives}</span></div>${team(x.team,'Team')}<div class="section-title">Shop offers</div><div class="shop"><div class="offers">${x.shopPets.map(petCard).join('')}</div><div class="foods">${x.shopFoods.map(foodCard).join('')}</div></div>`}
function render(){if(!DATA.slides.length){stage.innerHTML='<h2>No timeline frames</h2>';return}const s=DATA.slides[index];stage.innerHTML=s.type==='battle'?battle(s):shop(s);range.value=index;document.querySelector('#counter').textContent=`${index+1} / ${DATA.slides.length}`;document.querySelector('#prev').disabled=index===0;document.querySelector('#next').disabled=index===DATA.slides.length-1;}
document.querySelector('#prev').onclick=()=>{index=Math.max(0,index-1);render()};document.querySelector('#next').onclick=()=>{index=Math.min(DATA.slides.length-1,index+1);render()};range.oninput=e=>{index=Number(e.target.value);render()};document.addEventListener('keydown',e=>{if(e.key==='ArrowLeft')document.querySelector('#prev').click();if(e.key==='ArrowRight')document.querySelector('#next').click()});render();</script></body></html>'''
