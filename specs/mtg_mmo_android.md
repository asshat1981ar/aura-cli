# Arcane Realms: Magic-Style Card MMO for Android

## Summary

Build a Magic: The Gathering-style massively multiplayer online card game for Android. Players collect cards, build decks, battle other players in real-time duels, join guilds, participate in tournaments, and explore a rich fantasy world. The game features high-quality 2D/3D card art, spell effects, animated battlefields, and a polished UI with particle systems and shader effects.

## Workstream 1: Core Game Engine & Architecture [backend]

- Build the foundational Android project using Kotlin + Jetpack Compose for UI
- Implement the game state machine (lobby, deck builder, match, post-match)
- Create the core card data model: Card, Deck, Player, Match, Turn, Phase
- Build an entity-component system for card abilities and effects resolution
- Implement the mana system (5 colors + colorless, tapping, mana pool)
- Create the turn phase engine: Untap -> Upkeep -> Draw -> Main1 -> Combat -> Main2 -> End
- Build the stack/priority system for spell resolution and responses
- Add game rules engine with comprehensive MTG-style rule enforcement

Acceptance:
- [ ] Game state machine transitions are deterministic and tested
- [ ] Turn phases execute in correct order with priority passing
- [ ] Mana system supports all 5 colors plus colorless
- [ ] Stack resolves last-in-first-out correctly

## Workstream 2: Card System & Database [database]

- Create SQLite + Room database schema for card storage (500+ unique cards)
- Build card types: Creature, Instant, Sorcery, Enchantment, Artifact, Land, Planeswalker
- Implement card attributes: name, mana_cost, type_line, oracle_text, power, toughness, rarity
- Add keyword abilities system: Flying, Trample, Haste, Deathtouch, Lifelink, First Strike, etc.
- Build card collection and inventory management
- Create deck builder with validation (60 card minimum, 4-of rule, color identity)
- Implement card search, filter, and sort system
- Add card crafting/disenchanting economy (wildcards + dust)

Acceptance:
- [ ] Database stores 500+ cards with full attributes
- [ ] All 7 card types are supported with correct behavior
- [ ] 20+ keyword abilities are implemented
- [ ] Deck validation enforces all rules correctly

## Workstream 3: Graphics Engine & Visual Effects [frontend]

Depends on: Workstream 1

- Build the battlefield renderer using Android Canvas + OpenGL ES 3.0 for effects
- Implement card rendering with high-res art, frames, text overlay, and mana symbols
- Create animated card play sequences (hand -> stack -> battlefield transitions)
- Build particle systems for spell effects (fire, ice, lightning, nature, dark, light)
- Implement shader effects: card glow, holographic foil for rares/mythics, dissolve on destroy
- Add battlefield zones rendering: hand (fan layout), battlefield (grid), graveyard, exile, library
- Create attack/block animation system with creature combat visuals
- Build damage number popups and life total animations
- Implement day/night cycle and weather effects on battlefields
- Add haptic feedback for card plays, attacks, and spell resolution

Acceptance:
- [ ] Cards render at 60fps with art, frame, and text
- [ ] Particle effects for all 5 mana colors
- [ ] Smooth animations for all card state transitions
- [ ] Shader effects work on devices with OpenGL ES 3.0+

## Workstream 4: Multiplayer & Networking [backend] [api]

Depends on: Workstream 1, Workstream 2

- Build WebSocket-based real-time match server using Kotlin + Ktor
- Implement matchmaking system with ELO rating and rank tiers (Bronze through Mythic)
- Create game state synchronization with conflict resolution
- Build reconnection handling for dropped connections mid-match
- Implement anti-cheat: server-authoritative game state, move validation
- Add spectator mode for watching live matches
- Create tournament system: Swiss, single elimination, draft
- Build friend system with direct challenge capability
- Implement chat system with pre-built emotes and optional text

Acceptance:
- [ ] Match latency under 100ms for state updates
- [ ] Reconnection restores full game state within 5 seconds
- [ ] Server validates all moves; client cannot cheat
- [ ] Matchmaking finds opponents within 30 seconds

## Workstream 5: Player Progression & Economy [backend]

Depends on: Workstream 2

- Build player profile system: level, XP, rank, stats, match history
- Implement daily quest system (3 daily quests, weekly challenge)
- Create reward system: gold, gems (premium), card packs, wildcards
- Build the shop: card packs (5 cards each), cosmetics, battle passes
- Implement seasonal battle pass with free and premium tracks (100 levels)
- Add achievement system with 50+ achievements
- Create guild system: create, join, guild wars, shared rewards
- Build leaderboards: global, regional, guild, friends
- Implement draft mode economy (entry fee, keep-what-you-draft)

Acceptance:
- [ ] XP and leveling works with diminishing returns curve
- [ ] Daily quests refresh correctly at midnight UTC
- [ ] Shop transactions are atomic and rollback-safe
- [ ] Battle pass tracks progress across seasons

## Workstream 6: Card Art & Asset Pipeline [frontend]

- Build asset pipeline for card art generation and management
- Create 5 distinct card frame designs (one per mana color) + artifact/colorless/multi
- Implement procedural card art composition system using layered PNGs
- Build mana symbol icon set (W, U, B, R, G, C, X, hybrid symbols)
- Create battlefield background art: forest, mountain, swamp, island, plains + variants
- Build creature token art system for summoned tokens
- Implement loading screen art and splash screens
- Add animated card backs and sleeves (collectible cosmetics)
- Create UI icon set: menus, buttons, status indicators, rank badges

Acceptance:
- [ ] All 5 mana color card frames are visually distinct
- [ ] Card art renders crisp at 1080p and 1440p
- [ ] Asset loading is async with placeholder shimmer effect
- [ ] Total APK art assets under 200MB with on-demand download

## Workstream 7: AI Opponent System [backend]

Depends on: Workstream 1, Workstream 2

- Build AI opponent for single-player matches using MCTS (Monte Carlo Tree Search)
- Implement 4 difficulty levels: Beginner, Normal, Hard, Expert
- Create AI deck selection and mulligan strategy
- Build AI combat math: attack/block optimization
- Implement AI spell timing: when to counter, when to respond
- Add AI personality system (aggressive, control, midrange, combo)
- Create practice mode with AI hints for new players
- Build AI tournament opponents with increasing difficulty

Acceptance:
- [ ] Beginner AI makes intentional suboptimal plays
- [ ] Expert AI evaluates 3+ turns ahead via MCTS
- [ ] AI responds within 2 seconds per decision
- [ ] AI adapts strategy based on opponent's deck archetype

## Workstream 8: Audio System [frontend]

Depends on: Workstream 3

- Build audio engine with layered music system (ambient + battle intensity)
- Create sound effects for: card draw, card play, attack, damage, spell cast per color
- Implement dynamic music that intensifies during combat phases
- Add voice lines for legendary creatures and planeswalkers
- Build ambient soundscapes per battlefield theme
- Implement audio settings: master, music, SFX, voice volume sliders
- Create audio pooling system for concurrent sound effects
- Add spatial audio positioning for battlefield events

Acceptance:
- [ ] Music transitions smoothly between phases
- [ ] Sound effects are distinct per mana color
- [ ] Audio pooling handles 20+ simultaneous sounds
- [ ] Settings persist across sessions

## Workstream 9: UI/UX & Menu System [frontend]

Depends on: Workstream 3, Workstream 6

- Build main menu with animated card showcase background
- Create deck builder UI with drag-and-drop, mana curve display, color wheel
- Implement collection viewer with card gallery, filters, and zoom
- Build match HUD: life totals, mana pool, phase indicator, timer, hand count
- Create smooth hand fan with card peek, drag-to-play gesture
- Build settings screens: gameplay, audio, graphics quality, account
- Implement tutorial system with 5-chapter guided introduction
- Add notification system for quests, rewards, friend challenges
- Create social hub: friends list, guild panel, chat, spectate button

Acceptance:
- [ ] All screens support dark theme
- [ ] Card zoom shows full art and oracle text
- [ ] Drag-and-drop targeting works reliably on all screen sizes
- [ ] Tutorial completes in under 15 minutes

## Workstream 10: DevOps & Backend Infrastructure [infrastructure]

Depends on: Workstream 1

- Build CI/CD pipeline with GitHub Actions for Android builds
- Create Dockerized game server with auto-scaling via Kubernetes
- Implement monitoring: Prometheus metrics, Grafana dashboards
- Build database migration system for card balance patches
- Create A/B testing framework for economy tuning
- Implement analytics pipeline: match outcomes, card win rates, session length
- Build crash reporting integration (Firebase Crashlytics)
- Add feature flag system for gradual rollouts
- Create automated load testing for matchmaking server (10k concurrent)

Acceptance:
- [ ] CI builds and tests run in under 10 minutes
- [ ] Server auto-scales at 70% CPU threshold
- [ ] Analytics pipeline processes events within 1 minute
- [ ] Load test confirms 10k concurrent matches stable
