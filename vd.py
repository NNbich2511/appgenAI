"""from collections import Counter
import re


def vd( text, topn =3):
    word = re.findall(r'\b\w+\b', text.lower())
    count = Counter(word)
    return count.most_common(topn)

text = input("input: ")

out = vd(text)
for word, count in out:
    print(word, count)"""

"""from collections import deque


def shortest_path(grid):
    if not grid or grid[0][0] == 1:
        return -1

    n, m = len(grid), len(grid[0])
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # phải, xuống, trái, lên
    visited = [[False] * m for _ in range(n)]

    queue = deque([(0, 0, 1)])  # (x, y, path_length)
    visited[0][0] = True

    while queue:
        x, y, dist = queue.popleft()
        if x == n - 1 and y == m - 1:
            return dist
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < m and not visited[nx][ny] and grid[nx][ny] == 0:
                visited[nx][ny] = True
                queue.append((nx, ny, dist + 1))

    return -1


# Test
grid = [
    [0, 0, 1],
    [1, 0, 0],
    [1, 1, 0]
]
print(shortest_path(grid))"""


def vd(list):
    for i in range(len(list)//2):
        x = list[i]
        list[i] = list[len(list) - 1 - i]
        list[len(list) - 1 - i] = x

    return list

input = input("list:")
numbers = [int(x) for x in input.split()]

print(vd(numbers))