# Stack-specific demo-GIF guidelines (for our stacks)

> Internal playbook. **What's worth showing in a walkthrough depends on the
> architecture** — a single-cursor screen-capture flatters a single-user data app
> but completely misses what makes a live-collaborative app special. This maps each
> stack we build on to the distinctive runtime behaviour worth filming, the SDK
> primitives that actually produce *capturable motion*, and how to drive this tool
> to capture it. Grounded in the latest Streamlit and Convex‑React docs.

## TL;DR — choose the capture pattern by stack

| Stack | We use it for | The thing a demo MUST show | Capture pattern |
|---|---|---|---|
| **Streamlit** | single-user MVP, data tasks, workflow automation | the rerun model, **`st.write_stream`** token streaming, `st.status`/`st.fragment` live updates | **single cursor, single pane**; `burst` the stream + spinner |
| **Convex + React** | **live, multi-user collaboration** | a change in Client A appearing **live in Client B**; optimistic→atomic swap; presence/locks; agent mutations broadcast to all | **multi-pane** (2+ browser contexts side-by-side); `burst` the reactive propagation |
| **Next.js + SQL on Vercel** (Neon/PlanetScale; the *ideaflow / MewAgent* stack) | traditional request/response apps + streamed agents | RSC/Suspense streaming, Server Actions + `useOptimistic`, SSE agent stream | **single cursor**; `burst` the SSE/skeleton |

The single most important consequence: **Streamlit and Next.js single-user flows fit
this tool's default single-cursor capture; Convex collaboration needs a multi-pane
capture** (two synchronized browser contexts) — see the Convex section.

---

## Streamlit — single-user, quick MVP for data & workflow automation

**Why we reach for it:** one user, fast to build, perfect for data tasks and workflow
automation MVPs. There is exactly one session/cursor, so the default capture is ideal.

**What to show (and the primitive that makes it capturable):**
- **Token-by-token streaming** — `st.write_stream(stream)` renders a generator/LLM
  stream with a typewriter effect. This is the Streamlit answer to "show results
  *coming out*": `burst`-capture it and you get genuine streaming text, not a frozen
  spinner. (Chat pattern: `with st.chat_message("assistant"): response = st.write_stream(stream)`.)
- **Long-running work** — wrap in `with st.spinner("…")` or `st.status(...)`; `burst`
  it right after the click so the spinner/“running” state is real motion.
- **Live / auto-updating panels** — `@st.fragment(run_every="…")` reruns a fragment on
  an interval (live charts, background-job monitors) without a full rerun. `burst` the
  fragment region to show it ticking. Pair `run_every` with session state to start/stop.
- **The rerun model itself** — interacting reruns the script top-to-bottom; a clean
  walkthrough makes that legible (empty → input → rerun → result).

**Capture recipe:** one cursor; spec each step as `cap`/`act`; put a `burst` on the
`st.write_stream`/`st.status` phase; `scrollEl` the result widget. Streamlit-specific
gotchas (already in `SKILL.md`): scope locators to the **active tab panel**, await
upload registration before clicking, `st.data_editor` grids are **canvas** (wait on a
KPI/heading, not cell text), and capture the loading state on purpose.

> **To make a Streamlit demo show streaming results, the app must use
> `st.write_stream` / `st.fragment(run_every)`.** If it `await`s and renders atomically
> (like a plain `gemini_generate_*` call), the `burst` can only show the spinner — so
> stream in the app first, then capture.

*Sources: Streamlit docs — `st.write_stream`, `st.fragment(run_every)`, `st.spinner`/`st.status`, `st.rerun`, "Build a chat app" streaming pattern.*

---

## Convex + React — live, multi-user collaboration

**Why we reach for it:** real-time, multi-user collaboration. State is **server-led**:
clients subscribe via WebSockets and re-render reactively, so the interesting behaviour
is *cross-client* — which a single-pane recording cannot show.

**What to show (these are the whole point):**
1. **Reactive multi-client sync.** `useQuery(api.x.list, args)` is a live subscription —
   it re-renders every client whenever the server data changes. Film **two clients side
   by side**: Client A mutates; Client B's view updates live (no refresh). This is the
   headline.
2. **Optimistic update → atomic swap (zero flicker).**
   `useMutation(api.x.create).withOptimisticUpdate((localStore, args) => …)` paints the
   change locally at 0 ms (often a `crypto.randomUUID()` temp id via `insertAtTop` /
   `insertAtPosition` / `optimisticallyUpdateValueInPaginatedQuery`), then the server
   commit swaps the temp row for the real one in a single microtask — the VDOM diff is
   net-zero, so there's no flicker. `burst`-capture the temp→real transition. *(This is
   exactly the "optimistic overlay → authoritative base layer" reconciliation in our
   runtime diagram.)*
3. **Presence & locks.** Show collaborator cursors / “🔒 Locked by Agent” lease badges
   appearing and clearing across clients (our lease-lock trace) — the visible proof of
   coordination.
4. **Server-led agent, broadcast to everyone.** A `kickoffAgentJob` mutation → scheduled
   action → `internalMutation` commits that **broadcast down every client's WebSocket
   simultaneously**; `optimisticallySendMessage(api.x.listThreadMessages)` streams the
   agent's message into all panes. `burst` it to show the agent's output landing live
   for *both* users (contrast the old client-led SSE, which only Client 1 could see).

**Capture recipe (multi-pane — a documented extension of this tool):**
- Open **two `browser.newContext()`** sessions on the same workspace URL, set equal
  viewports, and tile them (A left, B right) — capture a frame spanning both.
- Drive the action in pane A; **`burst` while both panes are visible** so the reactive
  propagation (and the optimistic→swap, and lock appear/clear) is captured as motion.
- Render with a **side-by-side 2-up layout** + a "Client A / Client B" label per pane
  (the current `Walkthrough.jsx` is single-pane; collaboration needs a 2-up variant —
  tracked as the next capture mode).
- Caption the *cross-client* effect ("Client B sees it instantly — no refresh"), not the click.

> Single-cursor capture **undersells Convex**. The value is the second pane reacting.
> If you only have one pane, you've effectively filmed it as if it were Streamlit.

*Sources: Convex docs — `useQuery` (reactive subscription), `useMutation().withOptimisticUpdate` + `insertAtTop`/`insertAtPosition`/`optimisticallyUpdateValueInPaginatedQuery`, `optimisticallySendMessage`, `ConvexClient.onUpdate`; plus our MewAgent→NodeAgent runtime diagrams (optimistic overlay→atomic swap, lease leases, multi-client broadcast).*

---

## Next.js + SQL on Vercel — traditional request/response (the *ideaflow / MewAgent* stack)

**Why we reach for it:** conventional apps with a SQL system of record (Neon /
PlanetScale) on Vercel; also our original **MewAgent** stack — a Next.js API route that
returns a `ReadableStream` (SSE) the client applies to a MobX store.

**What to show:**
- **RSC / Suspense streaming** — route transition with a `loading.tsx` skeleton, then
  content streaming in. `burst` the skeleton→content swap.
- **Server Actions + `useOptimistic`** — optimistic UI on submit, then server reconcile;
  `burst` the optimistic-paint→confirmed transition (single-client analogue of Convex's swap).
- **SSE agent stream** — the MewAgent pattern: API route emits `type:"thought"` /
  `type:"client_action"` events; the client applies them live (nodes appearing as the
  agent works). `burst` the SSE event stream so the viewer watches the agent build the doc.
- **DB-backed forms** — create/edit round-tripping to Neon/PlanetScale.

**Capture recipe:** single cursor; `burst` the SSE/streaming/skeleton phase; `scrollEl`
the result. **Caveat worth showing (or noting):** client-led SSE means **multi-user
blindness** — a second browser on the same thread sees nothing until reload (our
diagram's "100% data blindness"). That's the contrast that motivates Convex; don't
fake multi-user on this stack.

*Sources: Next.js docs — App Router streaming/`loading.js` + Suspense, Server Actions, `useOptimistic`; our MewAgent client-led SSE runtime diagram.*

---

## What to `burst`, by stack (the "show the process" cheat-sheet)

| Stack | Burst this | Result the viewer sees |
|---|---|---|
| Streamlit | `st.write_stream` / `st.status` / `st.fragment` region | tokens typing out / spinner / live chart ticking |
| Convex + React | both panes during a mutation + agent commit | Client B updating live; optimistic temp→real swap; locks clearing |
| Next.js + Vercel | the SSE event stream / Suspense skeleton | agent nodes appearing; skeleton→content |

**Rule of thumb:** a demo can only show what the app renders. To show *streaming
results* you need a streaming primitive in the app (`st.write_stream`, Convex reactive
`useQuery` + `optimisticallySendMessage`, or SSE/RSC streaming) — then put a `burst`
right after the trigger. No streaming primitive ⇒ the honest best is the loading spinner.
