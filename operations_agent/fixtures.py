"""Seed fixtures: a realistic-ish backlog with deliberate near-duplicate clusters.

These exist so duplicate detection has something to match against (a fresh Jira
sandbox is empty) and so the demo collisions are guaranteed, not hoped for. Three
clusters are intentionally paraphrased duplicates of each other; the rest are
distinct bugs that should NOT match.

Each fixture is (summary, description, component). Keys are assigned by Jira /
the index when seeded.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FixtureBug:
    summary: str
    description: str
    component: str


# Cluster A — login crash (paraphrases of one bug)
# Cluster B — slow dashboard (paraphrases of one bug)
# Cluster C — email not sending (paraphrases of one bug)
SEED_FIXTURES: list[FixtureBug] = [
    # --- Cluster A: login/auth crash ---
    FixtureBug(
        "App crashes when logging in",
        "The application crashes to a white screen immediately after the user "
        "submits valid login credentials. Happens on every login attempt.",
        "auth",
    ),
    FixtureBug(
        "Login screen freezes and dies on submit",
        "After entering username and password and pressing enter, the login "
        "page hangs for a few seconds then the whole app dies.",
        "auth",
    ),
    FixtureBug(
        "Cannot sign in — application closes unexpectedly",
        "Signing in causes the app to close unexpectedly. Repro: open app, "
        "enter credentials, tap sign in, app vanishes.",
        "auth",
    ),
    # --- Cluster B: slow dashboard ---
    FixtureBug(
        "Dashboard takes forever to load",
        "The analytics dashboard takes 20+ seconds to render the charts after "
        "navigating to it. Spinner shows the entire time.",
        "dashboard",
    ),
    FixtureBug(
        "Reports page is extremely slow",
        "Loading the reports/analytics view is painfully slow — charts appear "
        "only after a long delay, sometimes half a minute.",
        "dashboard",
    ),
    # --- Cluster C: email not sending ---
    FixtureBug(
        "Password reset emails are not being delivered",
        "Users request a password reset but never receive the email. No bounce, "
        "nothing in spam. Reset link therefore never arrives.",
        "notifications",
    ),
    FixtureBug(
        "Notification emails fail to send",
        "Outbound notification emails (including reset links) are silently not "
        "sent. Queue shows them as processed but recipients get nothing.",
        "notifications",
    ),
    # --- Distinct, should NOT match the above ---
    FixtureBug(
        "Export to CSV produces empty file",
        "Exporting a table to CSV downloads a file with only headers and no "
        "rows, even when the table clearly has data.",
        "export",
    ),
    FixtureBug(
        "Profile avatar upload rejects PNG files",
        "Uploading a PNG as a profile picture returns 'unsupported file type', "
        "though PNG should be supported.",
        "profile",
    ),
    FixtureBug(
        "Search returns results in wrong order",
        "Full-text search results are not sorted by relevance; older, less "
        "relevant items appear above better matches.",
        "search",
    ),
    FixtureBug(
        "Dark mode toggle does not persist",
        "Enabling dark mode works for the session but resets to light mode "
        "after refreshing the page or reopening the app.",
        "settings",
    ),
    FixtureBug(
        "Pagination skips the last page of results",
        "When paginating a long list, the final page is unreachable — the Next "
        "button disables one page early.",
        "search",
    ),
    FixtureBug(
        "Timezone shown incorrectly on calendar events",
        "Calendar events display in UTC regardless of the user's configured "
        "timezone, making times off by several hours.",
        "calendar",
    ),
    FixtureBug(
        "File attachments over 5MB fail silently",
        "Attaching a file larger than 5MB appears to succeed but the attachment "
        "is missing when the message is sent.",
        "messaging",
    ),
    FixtureBug(
        "Webhook deliveries retry indefinitely on 500",
        "When a customer endpoint returns 500, our webhook system retries "
        "forever instead of backing off and giving up.",
        "integrations",
    ),
]
