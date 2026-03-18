import unittest

from scripts.merge_queue import MergeQueueManager, PullRequestState


class TestMergeQueueHelpers(unittest.TestCase):
    def setUp(self):
        self.manager = MergeQueueManager("owner/repo", "token", execute=False)

    def _pr(self, number, updated_at, diff, changed_files, deps=None):
        return PullRequestState(
            number=number,
            title=f"PR {number}",
            head_ref=f"branch-{number}",
            base_ref="main",
            updated_at=updated_at,
            additions=diff,
            deletions=0,
            changed_files=changed_files,
            draft=False,
            mergeable=True,
            mergeable_state="clean",
            requested_reviewers=0,
            dependency_prs=set(deps or []),
            approvals=1,
            checks_passed=True,
        )

    def test_mergeability_label_mapping(self):
        self.assertEqual(self.manager._mergeability_label(True, "clean", False), "clean")
        self.assertEqual(self.manager._mergeability_label(False, "dirty", False), "dirty")
        self.assertEqual(self.manager._mergeability_label(None, "blocked", False), "blocked")
        self.assertEqual(self.manager._mergeability_label(True, "unstable", False), "unstable")
        self.assertEqual(self.manager._mergeability_label(True, "clean", True), "blocked")

    def test_dependency_aware_priority_sort(self):
        states = {
            10: self._pr(10, "2026-03-10T00:00:00Z", diff=10, changed_files=3),
            11: self._pr(11, "2026-03-09T00:00:00Z", diff=5, changed_files=2, deps=[10]),
            12: self._pr(12, "2026-03-08T00:00:00Z", diff=2, changed_files=30),
            13: self._pr(13, "2026-03-07T00:00:00Z", diff=1, changed_files=2),
        }

        ordered = self.manager._topo_with_priority(states)
        numbers = [pr.number for pr in ordered]

        # Low-risk PR #13 leads by size/age, then #10, then dependent #11, then high-risk #12.
        self.assertEqual(numbers, [13, 10, 11, 12])

    def test_eligible_requires_clean_checks_and_approval(self):
        clean = self._pr(1, "2026-03-10T00:00:00Z", diff=1, changed_files=1)
        clean.mergeability = "clean"
        self.assertTrue(self.manager._eligible(clean))

        dirty = self._pr(2, "2026-03-10T00:00:00Z", diff=1, changed_files=1)
        dirty.mergeability = "dirty"
        self.assertFalse(self.manager._eligible(dirty))

        no_checks = self._pr(3, "2026-03-10T00:00:00Z", diff=1, changed_files=1)
        no_checks.mergeability = "clean"
        no_checks.checks_passed = False
        self.assertFalse(self.manager._eligible(no_checks))

        no_approval = self._pr(4, "2026-03-10T00:00:00Z", diff=1, changed_files=1)
        no_approval.mergeability = "clean"
        no_approval.approvals = 0
        self.assertFalse(self.manager._eligible(no_approval))


if __name__ == "__main__":
    unittest.main()
