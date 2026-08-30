import subprocess
import unittest

from homr.visual_sidecar.models import UPSTREAM_BASE_VERSION

UPSTREAM_REMOTE_URL = "https://github.com/liebharc/homr"
UPSTREAM_TAG_GLOB = "v[0-9]*"


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )


class TestUpstreamBaseVersion(unittest.TestCase):
    """UPSTREAM_BASE_VERSION must name the upstream release this fork actually branches from.

    The constant is hand-maintained, so it rots silently the moment someone rebases onto a
    newer upstream tag without updating it. Every visual sidecar this fork writes reports
    that constant as provenance, so a stale value is a false claim in a published artifact.
    """

    def test_upstream_base_version_matches_the_actual_fork_point(self) -> None:
        work_tree = _git("rev-parse", "--is-inside-work-tree")
        if work_tree.returncode != 0 or work_tree.stdout.strip() != "true":
            self.skipTest("not running from a git checkout")

        # The most recent upstream release reachable from HEAD is the fork point. A merge-base
        # check would not do: every older upstream tag is also an ancestor of HEAD, so it would
        # accept any release in the fork's history. Fork releases are tagged under "visual/" so
        # this glob matches upstream tags only.
        describe = _git("describe", "--tags", "--abbrev=0", "--match", UPSTREAM_TAG_GLOB, "HEAD")
        self.assertEqual(
            describe.returncode,
            0,
            "No upstream release tag is reachable from HEAD, so the fork point named by "
            "UPSTREAM_BASE_VERSION cannot be verified. Fetch the upstream tags with: "
            f"git remote add upstream {UPSTREAM_REMOTE_URL} && git fetch upstream --tags",
        )

        self.assertEqual(
            describe.stdout.strip(),
            f"v{UPSTREAM_BASE_VERSION}",
            "UPSTREAM_BASE_VERSION does not name the upstream release this branch is based "
            "on. Update it to the tag reported here, or rebase onto the release it claims.",
        )
