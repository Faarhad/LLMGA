import unittest

from desim.framework.queue import PriorityQueue


class TestPriorityQueue(unittest.TestCase):
    def test_priority_order(self) -> None:
        pq = PriorityQueue()
        pq.push(5, "late")
        pq.push(1, "early")
        self.assertEqual(pq.pop(), "early")
        self.assertEqual(pq.pop(), "late")

    def test_stable_same_priority(self) -> None:
        pq = PriorityQueue()
        pq.push(1, "a")
        pq.push(1, "b")
        self.assertEqual(pq.pop(), "a")
        self.assertEqual(pq.pop(), "b")

    def test_peek(self) -> None:
        pq = PriorityQueue()
        pq.push(2, "x")
        self.assertEqual(pq.peek(), "x")
        self.assertEqual(len(pq), 1)

    def test_pop_empty_raises(self) -> None:
        pq = PriorityQueue()
        with self.assertRaises(IndexError):
            pq.pop()


if __name__ == "__main__":
    unittest.main()

