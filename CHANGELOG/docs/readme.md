# README configuration recipe

## 2026-10-15 — Keep provider keys at the scope the app reads
A developer copying the secrets example placed all three feature API keys inside
the Supabase table, although the application reads them at the top level. Move
the unchanged placeholder assignments before `[supabase]`, matching the existing
setup prose and accessor contract. Storage/auth assignments remain in that table.

The same public-example parser and unchanged accessor check moves from three
unavailable feature keys to three visible keys; missing-key and wrong-scope
controls still return unavailable. No real credentials, application runtime,
provider calls, dependencies or release claims changed.

**Commit**: `this commit`. **Author**: Codex.
