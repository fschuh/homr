import subprocess
import unittest

from homr.visual_sidecar.models import UPSTREAM_VERSION

UPSTREAM_REMOTE_URL = "https://github.com/liebharc/homr"
UPSTREAM_TAG_GLOB = "v[0-9]*"


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )


class TestUpstreamVersion(unittest.TestCase):
    """UPSTREAM_VERSION must name the most recent upstream release contained in this history.

    The constant is hand-maintained, so it rots silently the moment a newer upstream release
    is merged without updating it. Every visual sidecar this fork writes reports that constant
    as provenance, so a stale value is a false claim in a published artifact.
    """

    def test_upstream_version_matches_the_most_recent_contained_release(self) -> None:
        work_tree = _git("rev-parse", "--is-inside-work-tree")
        if work_tree.returncode != 0 or work_tree.stdout.strip() != "true":
            self.skipTest("not running from a git checkout")

        # git describe reports the reachable tag with the fewest commits missing from it. Older
        # upstream releases are ancestors of newer ones, so the newest contained release always
        # wins, and this stays correct as upstream merges accumulate. A merge-base check would
        # not work: every older upstream tag is also an ancestor and would be accepted. Fork
        # releases are tagged under "visual/" so this glob matches upstream tags only.
        describe = _git("describe", "--tags", "--abbrev=0", "--match", UPSTREAM_TAG_GLOB, "HEAD")
        self.assertEqual(
            describe.returncode,
            0,
            "No upstream release tag is reachable from HEAD, so the upstream release named by "
            "UPSTREAM_VERSION cannot be verified. Fetch the upstream tags with: "
            f"git remote add upstream {UPSTREAM_REMOTE_URL} && git fetch upstream --tags",
        )

        self.assertEqual(
            describe.stdout.strip(),
            f"v{UPSTREAM_VERSION}",
            "UPSTREAM_VERSION does not name the most recent upstream release contained in this "
            "history. Update it to the tag reported here, in the commit that merged it.",
        )
