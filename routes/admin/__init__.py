"""
Heart Disease Prediction - Administrator package
======================================
Everything only an administrator may do. The routes are grouped by subject
across several small modules instead of one long file:

    users.py        user accounts, roles, suspension, doctor creation
    predictions.py  oversight of every saved assessment
    data.py         dataset inspection and system analytics
    models.py       model performance and retraining
    audits.py       the activity log

Every module registers on the SAME `admin_bp` blueprint defined here, so
the endpoint names are unchanged: `admin.users`, `admin.change_role` and
the rest all still resolve exactly as before.

Every route is protected by @role_required("admin"). There is a single
administrator tier holding every privilege in the system, and every
state-changing action is written to the activity log.
"""

import logging
import threading

from flask import Blueprint, flash, redirect, url_for

import auth_db
from routes.security import current_user, ROLE_ADMIN

log = logging.getLogger(__name__)

admin_bp = Blueprint("admin", __name__, url_prefix="/admin")

# ── Background training state ─────────────────────────────────────
# A single dict guarded by a lock. The training thread writes to it and
# the page reads from it, so the browser can poll progress.
_training_state = {
    "running": False,
    "started_at": None,
    "finished_at": None,
    "status": "idle",          # idle | running | success | failed
    "message": "",
    "duration": None,
}
_training_lock = threading.Lock()


def _can_manage(target, action="modify"):
    """
    Decide whether the signed-in administrator may act on this account.

    There is a single Admin tier now, so an Admin may manage anyone -
    including other Admins. Two safety rules remain:

      1. Nobody may act on their own account, so an admin cannot
         accidentally ban or delete themselves.
      2. The LAST usable Admin is protected. Without this, deleting,
         banning or demoting the final administrator would leave the
         system with nobody able to manage it and no way back in.

    Returns: (allowed, reason)
    """
    actor = current_user()

    if target["id"] == actor["id"]:
        return False, "You cannot perform this action on your own account."

    # Guard the last administrator standing
    if target["role"] == ROLE_ADMIN and not target.get("is_banned"):
        try:
            remaining = auth_db.count_active_admins()
        except Exception as e:
            log.error(f"Could not count admins: {e}", exc_info=True)
            return False, "The account could not be checked. Please try again."

        if remaining <= 1:
            return False, (f"This is the only active administrator, so it cannot "
                           f"be {action}. Promote another user to Admin first.")

    return True, ""


# Imported last: each module attaches its routes to admin_bp above, so
# admin_bp must already exist by the time these run.
from routes.admin import users, predictions, data, models, audits   # noqa: E402,F401
