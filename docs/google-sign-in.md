# Google sign-in

The application code is written. What remains is credentials, and those are
yours to create and paste — an OAuth client secret is a credential like any
other, and it should not pass through a chat transcript.

Three steps, about five minutes.

---

## 1. Create the OAuth client in Google Cloud

[console.cloud.google.com](https://console.cloud.google.com) → select or create a project.

**APIs & Services → OAuth consent screen**

| Field | Value |
|-------|-------|
| User type | External |
| App name | Lumen |
| Support email | yours |
| Authorised domain | `supabase.co` |
| Scopes | `email`, `profile`, `openid` — the defaults, nothing more |

While the app is in *Testing*, only accounts listed under **Test users** can sign
in. Add your own address there, or hit **Publish app** once you are ready for
anyone to.

**APIs & Services → Credentials → Create credentials → OAuth client ID**

| Field | Value |
|-------|-------|
| Application type | Web application |
| Name | Lumen web |
| Authorised JavaScript origins | `http://localhost:3000` and your production origin |
| **Authorised redirect URI** | `https://ifehvkuddfetxfhtvmnw.supabase.co/auth/v1/callback` |

That redirect URI is the one that matters, and it is the one people get wrong.
It points at **Supabase**, not at your app — Supabase receives the callback from
Google and then redirects to you. A mismatch here produces
`Error 400: redirect_uri_mismatch`, and the fix is always to copy this string
exactly, including `https://` and no trailing slash.

Google gives you a **Client ID** and a **Client secret**.

---

## 2. Paste them into Supabase

Dashboard → **Authentication → Sign In / Providers → Google**

1. Toggle **Enable Sign in with Google**.
2. Paste the Client ID and Client secret.
3. Save.

Supabase shows its callback URL on that same screen — confirm it matches what
you entered in Google.

---

## 3. Allow your app's redirect

Dashboard → **Authentication → URL Configuration**

| Field | Value |
|-------|-------|
| Site URL | `http://localhost:3000` (your production origin once deployed) |
| Redirect URLs | `http://localhost:3000/auth/callback`, and the same path on production |

The app sends people to `/auth/callback?next=/app`. A redirect URL that is not
on this allow-list is silently dropped by Supabase and the person lands back on
the sign-in page with no error — which is a genuinely confusing failure, so it
is worth checking twice.

---

## While you are in there: turn off email confirmation for development

**Authentication → Sign In / Providers → Email → Confirm email: off**

The project currently has `mailer_autoconfirm: false`, so a new account cannot
act until someone opens a link in an inbox. That blocks the end-to-end test —
sign up, upload, get a proposal, accept — for a reason that has nothing to do
with the code. Turn confirmation back **on** before real users exist.

---

## What the code already does

- `apps/web/src/lib/supabase/client.ts` — browser client, PKCE flow, session persistence.
- `apps/web/src/lib/supabase/auth.ts` — `signInWithGoogle()`, `signUpWithEmail()`, `signInWithEmail()`, `signOut()`, `onAuthChange()`.
- Sign-up sends `display_name` and `org_name` as user metadata, which the
  `handle_new_user` trigger reads to create the profile, the organization and the
  owner membership in one transaction. Google sign-in carries no `org_name`, so
  the trigger falls back to `"<name>'s workspace"` — the person gets a working
  workspace on first sign-in either way, with no extra screen.
- `access_type=offline` and `prompt=select_account` are set for Google, so
  switching accounts shows the chooser instead of silently reusing the last one.

## Verifying it worked

After signing in with Google once, this should return one row:

```sql
select p.email, o.name as org, m.role
from public.profiles p
join public.memberships m on m.user_id = p.id
join public.organizations o on o.id = m.org_id
order by p.created_at desc
limit 1;
```

No row means the `handle_new_user` trigger did not fire — check
**Database → Triggers** for `on_auth_user_created` on `auth.users`.
