# 75-leetcode-algo

Here’s a **deep explanation with pseudo code**, **algorithm**, **time**, and **space complexity** for each of the 9 problems:

---

### **1. Merge Strings Alternately**

**Algorithm:**

* Use a pointer `i`, iterate up to the max length of both strings.
* Alternately pick characters from both strings.

**Pseudo Code:**

```pseudo
result = ""
i = 0
while i < max(length of word1, word2):
    if i < length of word1: result += word1[i]
    if i < length of word2: result += word2[i]
    i += 1
return result
```

**Time Complexity:** O(n + m)
**Space Complexity:** O(n + m) (new string)

---

### **2. Greatest Common Divisor of Strings**

**Algorithm:**

* If str1 + str2 != str2 + str1 → no common base string.
* Else, GCD of lengths gives the repeated unit.

**Pseudo Code:**

```pseudo
if str1 + str2 != str2 + str1:
    return ""
gcdLength = gcd(len(str1), len(str2))
return str1[0:gcdLength]
```

**Time Complexity:** O(n + m) (string concatenation check)
**Space Complexity:** O(n + m) (for comparison)

---

### **3. Kids With the Greatest Number of Candies**

**Algorithm:**

* First pass: get max in candies.
* Second pass: for each kid, check if their candies + extra ≥ max.

**Pseudo Code:**

```pseudo
maxVal = max(candies)
result = []
for candy in candies:
    result.append(candy + extra >= maxVal)
return result
```

**Time Complexity:** O(n)
**Space Complexity:** O(n)

---

### **4. Can Place Flowers**

**Algorithm:**

* Traverse the array, check if current and adjacent spots are empty.
* If yes, plant a flower and skip the next spot.

**Pseudo Code:**

```pseudo
count = 0
for i in 0 to n-1:
    if flowerbed[i] == 0 and (i == 0 or flowerbed[i-1] == 0) and (i == n-1 or flowerbed[i+1] == 0):
        flowerbed[i] = 1
        count += 1
return count >= n
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

---

### **5. Reverse Vowels of a String**

**Algorithm:**

* Use two pointers from both ends.
* Swap vowels when both pointers hit vowels.

**Pseudo Code:**

```pseudo
left = 0, right = len(s) - 1
while left < right:
    while s[left] not in vowels: left += 1
    while s[right] not in vowels: right -= 1
    swap s[left] and s[right]
    left += 1, right -= 1
```

**Time Complexity:** O(n)
**Space Complexity:** O(n) if new string, O(1) if in-place mutable structure

---

### **6. Reverse Words in a String**

**Algorithm:**

* Trim leading/trailing spaces.
* Split words, reverse the array, join with single space.

**Pseudo Code:**

```pseudo
words = split(trim(s), " ")
reverse(words)
return join(words, " ")
```

**Time Complexity:** O(n)
**Space Complexity:** O(n)

---

### **7. Product of Array Except Self**

**Algorithm:**

* Use prefix and suffix passes to multiply product except self.
* 1st pass left-to-right for prefix.
* 2nd pass right-to-left for suffix.

**Pseudo Code:**

```pseudo
res = [1] * n
prefix = 1
for i in 0 to n-1:
    res[i] = prefix
    prefix *= nums[i]
suffix = 1
for i in n-1 to 0:
    res[i] *= suffix
    suffix *= nums[i]
```

**Time Complexity:** O(n)
**Space Complexity:** O(1) (excluding output array)

---

### **8. Increasing Triplet Subsequence**

**Algorithm:**

* Track two smallest numbers: `first`, `second`.
* If you find a number > second → triplet exists.

**Pseudo Code:**

```pseudo
first = inf, second = inf
for num in nums:
    if num <= first: first = num
    else if num <= second: second = num
    else: return true
return false
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

---

### **9. String Compression**

**Algorithm:**

* Use two pointers: `i` to read, `write` to overwrite.
* Count repeats and write character + count.

**Pseudo Code:**

```pseudo
write = 0, i = 0
while i < len(chars):
    char = chars[i]
    count = 0
    while i < len(chars) and chars[i] == char:
        i += 1
        count += 1
    chars[write] = char
    write += 1
    if count > 1:
        for digit in str(count):
            chars[write] = digit
            write += 1
return write
```

**Time Complexity:** O(n)
**Space Complexity:** O(1) (in-place compression)

Here’s a **deep explanation with algorithm, pseudo code, time, and space complexity** for each **Two Pointers** problem:

---

### **1. Move Zeroes**

**🧠 Algorithm:**
Two pointers:

* `lastNonZeroIndex` tracks position to place non-zero.
* Traverse with `i`; if nums\[i] ≠ 0, swap with `lastNonZeroIndex`.

**🧾 Pseudo Code:**

```pseudo
last = 0
for i in 0 to n-1:
    if nums[i] ≠ 0:
        swap(nums[i], nums[last])
        last += 1
```

**⏱ Time Complexity:** O(n)
**📦 Space Complexity:** O(1) (in-place)

---

### **2. Is Subsequence**

**🧠 Algorithm:**
Use two pointers on `s` and `t`.
Move `s` pointer only if characters match.

**🧾 Pseudo Code:**

```pseudo
i = 0, j = 0
while i < len(s) and j < len(t):
    if s[i] == t[j]: i += 1
    j += 1
return i == len(s)
```

**⏱ Time Complexity:** O(n) where n = length of t
**📦 Space Complexity:** O(1)

---

### **3. Container With Most Water**

**🧠 Algorithm:**
Two pointers at ends.
Move the shorter height inward.
Track max area = min(height\[i], height\[j]) × (j - i)

**🧾 Pseudo Code:**

```pseudo
i = 0, j = n - 1, maxArea = 0
while i < j:
    area = min(height[i], height[j]) * (j - i)
    maxArea = max(maxArea, area)
    if height[i] < height[j]: i += 1
    else: j -= 1
```

**⏱ Time Complexity:** O(n)
**📦 Space Complexity:** O(1)

---

### **4. Max Number of K-Sum Pairs**

**🧠 Algorithm:**

* Sort array.
* Use two pointers: left + right == k → count++, move both.
* If sum < k → move left. If sum > k → move right.

**🧾 Pseudo Code:**

```pseudo
sort(nums)
left = 0, right = n - 1, count = 0
while left < right:
    sum = nums[left] + nums[right]
    if sum == k:
        count += 1
        left += 1
        right -= 1
    elif sum < k:
        left += 1
    else:
        right -= 1
```

**⏱ Time Complexity:** O(n log n) (due to sorting)
**📦 Space Complexity:** O(1)

Here’s a deep explanation for **Sliding Window** problems with **algorithm**, **pseudo code**, **time**, and **space complexity**:

---

### **1. Maximum Average Subarray I**

**🧠 Algorithm:**
Use a sliding window of size `k` and calculate sum.
Slide window by removing `nums[i - k]` and adding `nums[i]`.

**🧾 Pseudo Code:**

```pseudo
sum = sum of first k elements
maxSum = sum
for i = k to n-1:
    sum += nums[i] - nums[i - k]
    maxSum = max(maxSum, sum)
return maxSum / k
```

**⏱ Time Complexity:** O(n)
**📦 Space Complexity:** O(1)

---

### **2. Maximum Number of Vowels in a Substring of Given Length**

**🧠 Algorithm:**
Use sliding window of size `k`, count vowels.
Slide and update count based on entering and exiting char.

**🧾 Pseudo Code:**

```pseudo
count = number of vowels in first k chars
maxCount = count
for i = k to n-1:
    if s[i - k] is vowel: count -= 1
    if s[i] is vowel: count += 1
    maxCount = max(maxCount, count)
return maxCount
```

**⏱ Time Complexity:** O(n)
**📦 Space Complexity:** O(1)

---

### **3. Max Consecutive Ones III**

**🧠 Algorithm:**
Use sliding window and track number of zeros.
Shrink window from left when zeros > k.

**🧾 Pseudo Code:**

```pseudo
left = 0, zeros = 0, maxLen = 0
for right in 0 to n-1:
    if nums[right] == 0: zeros += 1
    while zeros > k:
        if nums[left] == 0: zeros -= 1
        left += 1
    maxLen = max(maxLen, right - left + 1)
return maxLen
```

**⏱ Time Complexity:** O(n)
**📦 Space Complexity:** O(1)

---

### **4. Longest Subarray of 1's After Deleting One Element**

**🧠 Algorithm:**
Sliding window to allow at most 1 zero.
Track window length, and shrink when >1 zero.
Answer is max window size - 1 (we must delete one element).

**🧾 Pseudo Code:**

```pseudo
left = 0, zeros = 0, maxLen = 0
for right in 0 to n-1:
    if nums[right] == 0: zeros += 1
    while zeros > 1:
        if nums[left] == 0: zeros -= 1
        left += 1
    maxLen = max(maxLen, right - left + 1)
return maxLen - 1
```

**⏱ Time Complexity:** O(n)
**📦 Space Complexity:** O(1)

Here’s a **detailed breakdown** for each problem under **Prefix Sum** and **Hash Map / Set** with:

* ✅ **Algorithm**
* 🧾 **Pseudo Code**
* ⏱ **Time Complexity**
* 📦 **Space Complexity**

---

## 🔢 Prefix Sum

---

### **1. Find the Highest Altitude**

**✅ Algorithm:**
Track cumulative sum and store max as we go.

**🧾 Pseudo Code:**

```pseudo
maxAlt = 0, current = 0
for gain in gains:
    current += gain
    maxAlt = max(maxAlt, current)
return maxAlt
```

**⏱ Time:** O(n)
**📦 Space:** O(1)

---

### **2. Find Pivot Index**

**✅ Algorithm:**
Use total sum and left sum:
At index `i`, if `leftSum == total - leftSum - nums[i]`, it's the pivot.

**🧾 Pseudo Code:**

```pseudo
total = sum(nums)
leftSum = 0
for i in 0 to n-1:
    if leftSum == total - leftSum - nums[i]: return i
    leftSum += nums[i]
return -1
```

**⏱ Time:** O(n)
**📦 Space:** O(1)

---

## 🧩 Hash Map / Set

---

### **3. Find the Difference of Two Arrays**

**✅ Algorithm:**
Convert to sets, subtract to find differences.

**🧾 Pseudo Code:**

```pseudo
set1 = set(nums1)
set2 = set(nums2)
return [list(set1 - set2), list(set2 - set1)]
```

**⏱ Time:** O(n + m)
**📦 Space:** O(n + m)

---

### **4. Unique Number of Occurrences**

**✅ Algorithm:**
Use hashmap for frequency count, set to check uniqueness.

**🧾 Pseudo Code:**

```pseudo
freq = {}
for num in arr: freq[num] += 1
return len(set(freq.values())) == len(freq.values())
```

**⏱ Time:** O(n)
**📦 Space:** O(n)

---

### **5. Determine if Two Strings Are Close**

**✅ Algorithm:**

* Both strings must have same set of chars.
* Char frequencies must be equal in count (but not order).

**🧾 Pseudo Code:**

```pseudo
if set(s1) != set(s2): return False
if sorted(freq(s1)) != sorted(freq(s2)): return False
return True
```

**⏱ Time:** O(n log n)
**📦 Space:** O(n)

---

### **6. Equal Row and Column Pairs**

**✅ Algorithm:**

* Convert each row into a tuple and count frequency.
* For each column (as tuple), check if it exists in row map.

**🧾 Pseudo Code:**

```pseudo
count = hashmap of row tuples
for each column:
    convert to tuple and check count in row map
    add count to result
return result
```

**⏱ Time:** O(n²)
**📦 Space:** O(n²)

Here’s a **deep explanation** for each **Stack** problem with:

* ✅ **Algorithm**
* 🧾 **Pseudo Code**
* ⏱ **Time Complexity**
* 📦 **Space Complexity**

---

## 🧱 Stack

---

### **1. Removing Stars From a String**

**✅ Algorithm:**
Use a stack. Traverse the string:

* If char is not `*`, push to stack.
* If char is `*`, pop the top of the stack.

**🧾 Pseudo Code:**

```pseudo
stack = []
for ch in s:
    if ch != '*': stack.push(ch)
    else if stack not empty: stack.pop()
return join(stack)
```

**⏱ Time:** O(n)
**📦 Space:** O(n)

---

### **2. Asteroid Collision**

**✅ Algorithm:**
Use a stack to simulate collisions:

* Push positive asteroids.
* For negative asteroids, compare and pop smaller positive ones.
* If equal, destroy both.

**🧾 Pseudo Code:**

```pseudo
stack = []
for asteroid in asteroids:
    while stack not empty and asteroid < 0 < stack.top():
        if abs(asteroid) > stack.top(): stack.pop()
        elif abs(asteroid) == stack.top(): stack.pop(); break
        else: break
    else:
        stack.push(asteroid)
return stack
```

**⏱ Time:** O(n)
**📦 Space:** O(n)

---

### **3. Decode String**

**✅ Algorithm:**
Use two stacks: one for numbers, one for strings.

* Push current string and number on seeing `[`.
* Build substring until `]` and repeat it.

**🧾 Pseudo Code:**

```pseudo
numStack = []
strStack = []
currentStr = "", num = 0
for ch in s:
    if digit: num = num * 10 + int(ch)
    elif ch == '[': 
        numStack.push(num)
        strStack.push(currentStr)
        num = 0
        currentStr = ""
    elif ch == ']':
        repeat = numStack.pop()
        prevStr = strStack.pop()
        currentStr = prevStr + currentStr * repeat
    else:
        currentStr += ch
return currentStr
```

**⏱ Time:** O(n)
**📦 Space:** O(n)

Here’s a **deep explanation** of the **Queue**-based problems with:

* ✅ **Algorithm**
* 🧾 **Pseudo Code**
* ⏱ **Time Complexity**
* 📦 **Space Complexity**

---

## 📥 Queue

---

### **1. Number of Recent Calls**

**✅ Algorithm:**
Maintain a queue of timestamps.
For each `ping(t)`, add `t` to queue and remove all timestamps `< t - 3000`.

**🧾 Pseudo Code:**

```pseudo
queue = []

function ping(t):
    queue.push(t)
    while queue.front() < t - 3000:
        queue.pop()
    return queue.length
```

**⏱ Time Complexity:** Amortized O(1) per ping
**📦 Space Complexity:** O(n), where n is number of pings in 3000ms window

---

### **2. Dota2 Senate**

**✅ Algorithm:**
Use two queues: one for Radiant, one for Dire.
Simulate banning in round-robin fashion:

* Compare front of both queues.
* The earlier index wins, bans opponent, and re-queues to `i + n`.

**🧾 Pseudo Code:**

```pseudo
radiantQueue = indices of 'R'
direQueue = indices of 'D'
n = senate.length

while radiantQueue and direQueue:
    r = radiantQueue.pop()
    d = direQueue.pop()
    if r < d:
        radiantQueue.push(r + n)
    else:
        direQueue.push(d + n)

return "Radiant" if radiantQueue else "Dire"
```

**⏱ Time Complexity:** O(n)
**📦 Space Complexity:** O(n)


Here’s a **deep explanation** of the key **Linked List** problems with:

* ✅ **Algorithm**
* 🧾 **Pseudo Code**
* ⏱ **Time Complexity**
* 📦 **Space Complexity**

---

## 🔗 Linked List

---

### **1. Delete the Middle Node of a Linked List**

**✅ Algorithm:**
Use slow and fast pointers to find the middle.
Track previous node of slow to remove it.

**🧾 Pseudo Code:**

```pseudo
if head.next == null: return null
slow = head, fast = head, prev = null
while fast and fast.next:
    prev = slow
    slow = slow.next
    fast = fast.next.next
prev.next = slow.next
return head
```

**⏱ Time:** O(n)
**📦 Space:** O(1)

---

### **2. Odd Even Linked List**

**✅ Algorithm:**
Separate nodes by odd/even indices.
Connect end of odd list to head of even list.

**🧾 Pseudo Code:**

```pseudo
if head == null: return null
odd = head, even = head.next, evenHead = even
while even and even.next:
    odd.next = even.next
    odd = odd.next
    even.next = odd.next
    even = even.next
odd.next = evenHead
return head
```

**⏱ Time:** O(n)
**📦 Space:** O(1)

---

### **3. Reverse Linked List**

**✅ Algorithm:**
Iterate and reverse `next` pointers using three pointers: `prev`, `curr`, `next`.

**🧾 Pseudo Code:**

```pseudo
prev = null, curr = head
while curr:
    next = curr.next
    curr.next = prev
    prev = curr
    curr = next
return prev
```

**⏱ Time:** O(n)
**📦 Space:** O(1)

---

### **4. Maximum Twin Sum of a Linked List**

**✅ Algorithm:**

* Use fast/slow to find middle.
* Reverse second half.
* Traverse both halves to compute max twin sum.

**🧾 Pseudo Code:**

```pseudo
slow = head, fast = head
while fast and fast.next:
    slow = slow.next
    fast = fast.next.next

second = reverse(slow)
first = head, maxSum = 0
while second:
    maxSum = max(maxSum, first.val + second.val)
    first = first.next
    second = second.next
return maxSum
```

**⏱ Time:** O(n)
**📦 Space:** O(1)

Here’s a deep explanation of **Binary Tree – DFS** problems with:

* ✅ **Algorithm**
* 🧾 **Pseudo Code**
* ⏱ **Time Complexity**
* 📦 **Space Complexity**

---

## 🌳 Binary Tree – DFS

---

### **1. Maximum Depth of Binary Tree**

**✅ Algorithm:**
Recursively compute depth of left and right subtrees.

**🧾 Pseudo Code:**

```pseudo
function maxDepth(node):
    if node == null: return 0
    return 1 + max(maxDepth(node.left), maxDepth(node.right))
```

**⏱ Time:** O(n)
**📦 Space:** O(h) (h = tree height)

---

### **2. Leaf-Similar Trees**

**✅ Algorithm:**
Use DFS to collect all leaf values in order from both trees, then compare.

**🧾 Pseudo Code:**

```pseudo
function getLeaves(node, list):
    if node == null: return
    if node.left == null and node.right == null:
        list.append(node.val)
    getLeaves(node.left, list)
    getLeaves(node.right, list)

leaves1 = [], leaves2 = []
getLeaves(root1, leaves1)
getLeaves(root2, leaves2)
return leaves1 == leaves2
```

**⏱ Time:** O(n + m)
**📦 Space:** O(n + m)

---

### **3. Count Good Nodes in Binary Tree**

**✅ Algorithm:**
DFS with `maxSoFar`. Count node if `node.val >= maxSoFar`.

**🧾 Pseudo Code:**

```pseudo
function dfs(node, maxSoFar):
    if node == null: return 0
    count = 1 if node.val >= maxSoFar else 0
    maxSoFar = max(maxSoFar, node.val)
    count += dfs(node.left, maxSoFar)
    count += dfs(node.right, maxSoFar)
    return count

return dfs(root, root.val)
```

**⏱ Time:** O(n)
**📦 Space:** O(h)

---

### **4. Path Sum III**

**✅ Algorithm:**

* For each node, start a DFS to count paths that sum to target.
* Use helper to count paths from that node.

**🧾 Pseudo Code:**

```pseudo
function pathSum(root, target):
    if root == null: return 0
    return count(root, target) + pathSum(root.left, target) + pathSum(root.right, target)

function count(node, sum):
    if node == null: return 0
    sum -= node.val
    return (1 if sum == 0 else 0) + count(node.left, sum) + count(node.right, sum)
```

**⏱ Time:** O(n²) worst case
**📦 Space:** O(h)

---

### **5. Longest ZigZag Path in a Binary Tree**

**✅ Algorithm:**
DFS with direction and length tracking:

* Left → Right
* Right → Left

**🧾 Pseudo Code:**

```pseudo
maxLen = 0
function dfs(node, isLeft, length):
    if node == null: return
    maxLen = max(maxLen, length)
    if isLeft:
        dfs(node.left, false, 1)
        dfs(node.right, true, length + 1)
    else:
        dfs(node.left, false, length + 1)
        dfs(node.right, true, 1)

dfs(root, true, 0)
dfs(root, false, 0)
return maxLen
```

**⏱ Time:** O(n)
**📦 Space:** O(h)

---

### **6. Lowest Common Ancestor of a Binary Tree**

**✅ Algorithm:**
Recursive DFS.
If one node in left subtree, one in right → current is LCA.

**🧾 Pseudo Code:**

```pseudo
function LCA(root, p, q):
    if root == null or root == p or root == q: return root
    left = LCA(root.left, p, q)
    right = LCA(root.right, p, q)
    if left and right: return root
    return left if left else right
```

**⏱ Time:** O(n)
**📦 Space:** O(h)

Here’s a **deep explanation** of **Binary Tree – BFS** and **Binary Search Tree (BST)** problems with:

* ✅ **Algorithm**
* 🧾 **Pseudo Code**
* ⏱ **Time Complexity**
* 📦 **Space Complexity**

---

## 🌳 Binary Tree – BFS

---

### **1. Binary Tree Right Side View**

**✅ Algorithm:**
Level-order traversal (BFS), record the last node of each level.

**🧾 Pseudo Code:**

```pseudo
queue = [root]
result = []
while queue not empty:
    size = len(queue)
    for i in 0 to size - 1:
        node = queue.pop()
        if i == size - 1: result.append(node.val)
        if node.left: queue.push(node.left)
        if node.right: queue.push(node.right)
return result
```

**⏱ Time:** O(n)
**📦 Space:** O(w), width of tree

---

### **2. Maximum Level Sum of a Binary Tree**

**✅ Algorithm:**
Do BFS, compute sum of each level, track max.

**🧾 Pseudo Code:**

```pseudo
queue = [root], level = 1, maxLevel = 1, maxSum = -∞
while queue not empty:
    size = len(queue), sum = 0
    for i in 0 to size-1:
        node = queue.pop()
        sum += node.val
        if node.left: queue.push(node.left)
        if node.right: queue.push(node.right)
    if sum > maxSum:
        maxSum = sum
        maxLevel = level
    level += 1
return maxLevel
```

**⏱ Time:** O(n)
**📦 Space:** O(w)

---

## 🌲 Binary Search Tree (BST)

---

### **3. Search in a Binary Search Tree**

**✅ Algorithm:**
If node is null or matches value → return.
Else move left or right based on comparison.

**🧾 Pseudo Code:**

```pseudo
function searchBST(root, val):
    if root == null or root.val == val: return root
    if val < root.val: return searchBST(root.left, val)
    return searchBST(root.right, val)
```

**⏱ Time:** O(h)
**📦 Space:** O(h)

---

### **4. Delete Node in a BST**

**✅ Algorithm:**

* Find node to delete.
* Case 1: No child → return null.
* Case 2: One child → return child.
* Case 3: Two children → replace with inorder successor (min of right subtree).

**🧾 Pseudo Code:**

```pseudo
function deleteNode(root, key):
    if root == null: return null
    if key < root.val: root.left = deleteNode(root.left, key)
    elif key > root.val: root.right = deleteNode(root.right, key)
    else:
        if root.left == null: return root.right
        if root.right == null: return root.left
        minLargerNode = findMin(root.right)
        root.val = minLargerNode.val
        root.right = deleteNode(root.right, minLargerNode.val)
    return root

function findMin(node):
    while node.left: node = node.left
    return node
```

**⏱ Time:** O(h)
**📦 Space:** O(h)

Here’s a **deep explanation** for key **Graph DFS & BFS problems** with:

* ✅ **Algorithm**
* 🧾 **Pseudo Code**
* ⏱ **Time Complexity**
* 📦 **Space Complexity**

---

## 🔗 Graphs – DFS

---

### **1. Keys and Rooms**

**✅ Algorithm:**
Use DFS from room 0. Mark visited rooms.
If all rooms visited → return true.

**🧾 Pseudo Code:**

```pseudo
visited = set()

function dfs(room):
    visited.add(room)
    for key in rooms[room]:
        if key not in visited:
            dfs(key)

dfs(0)
return len(visited) == len(rooms)
```

**⏱ Time:** O(n + e)
**📦 Space:** O(n)

---

### **2. Number of Provinces**

**✅ Algorithm:**
Use DFS on adjacency matrix.
Each DFS call from unvisited node is one province.

**🧾 Pseudo Code:**

```pseudo
visited = set()
count = 0
for i in 0 to n-1:
    if i not in visited:
        dfs(i)
        count += 1

function dfs(node):
    visited.add(node)
    for neighbor in 0 to n-1:
        if isConnected[node][neighbor] == 1 and neighbor not in visited:
            dfs(neighbor)

return count
```

**⏱ Time:** O(n²)
**📦 Space:** O(n)

---

### **3. Reorder Routes to Make All Paths Lead to the City Zero**

**✅ Algorithm:**

* Build a graph with direction info.
* DFS from node 0.
* If edge goes away from 0, count it for reversal.

**🧾 Pseudo Code:**

```pseudo
build graph with (neighbor, direction) where direction = 1 if original edge u → v

function dfs(node):
    visited.add(node)
    for neighbor, dir in graph[node]:
        if neighbor not in visited:
            count += dir
            dfs(neighbor)

dfs(0)
return count
```

**⏱ Time:** O(n)
**📦 Space:** O(n)

---

### **4. Evaluate Division**

**✅ Algorithm:**
Build weighted graph.
DFS with running product of weights from `start` to `end`.

**🧾 Pseudo Code:**

```pseudo
graph = buildGraph(equations, values)

function dfs(curr, target, product):
    if curr == target: return product
    visited.add(curr)
    for neighbor, value in graph[curr]:
        if neighbor not in visited:
            res = dfs(neighbor, target, product * value)
            if res != -1: return res
    return -1

for query in queries:
    if query[0] not in graph or query[1] not in graph: result = -1
    else: result = dfs(query[0], query[1], 1)
```

**⏱ Time:** O(n \* q)
**📦 Space:** O(n)

---

## 📥 Graphs – BFS

---

### **5. Nearest Exit from Entrance in Maze**

**✅ Algorithm:**
Use BFS from entrance.
Track visited cells and steps.
Return step when reaching a border cell ≠ entrance.

**🧾 Pseudo Code:**

```pseudo
queue = [(entrance, 0)]
visited = set(entrance)

while queue:
    (r, c), steps = queue.pop()
    if at border and not entrance: return steps
    for each neighbor in 4 directions:
        if in bounds and cell == '.' and not visited:
            queue.push((neighbor, steps + 1))
            visited.add(neighbor)
return -1
```

**⏱ Time:** O(m × n)
**📦 Space:** O(m × n)

---

### **6. Rotting Oranges**

**✅ Algorithm:**
Use multi-source BFS from all rotten oranges.
Each level represents a minute.

**🧾 Pseudo Code:**

```pseudo
queue = all (i, j) where grid[i][j] == 2
fresh = count of 1s
minutes = 0

while queue and fresh > 0:
    for all cells in current level:
        for each neighbor:
            if neighbor == 1:
                mark as rotten, fresh -= 1
                queue.push(neighbor)
    minutes += 1

return minutes if fresh == 0 else -1
```

**⏱ Time:** O(m × n)
**📦 Space:** O(m × n)

Here’s a **deep explanation** of **Heap / Priority Queue** problems with:

* ✅ **Algorithm**
* 🧾 **Pseudo Code**
* ⏱ **Time Complexity**
* 📦 **Space Complexity**

---

## 🥇 Heap / Priority Queue

---

### **1. Kth Largest Element in an Array**

**✅ Algorithm:**
Use **min-heap** of size `k`.

* Push first `k` elements.
* For remaining, if current > heap\[0], pop and push.

**🧾 Pseudo Code:**

```pseudo
heap = first k elements
heapify(heap)

for i in k to n-1:
    if nums[i] > heap[0]:
        pop heap
        push nums[i] to heap

return heap[0]
```

**⏱ Time:** O(n log k)
**📦 Space:** O(k)

---

### **2. Smallest Number in Infinite Set**

**✅ Algorithm:**

* Use `min-heap` and a set to track added-back numbers.
* `popSmallest()` pops from heap or next smallest.
* `addBack(num)` pushes back if removed before.

**🧾 Pseudo Code:**

```pseudo
next = 1
minHeap = []
inHeap = set()

function popSmallest():
    if heap not empty:
        x = heap.pop()
        inHeap.remove(x)
        return x
    else:
        next += 1
        return next - 1

function addBack(num):
    if num < next and num not in inHeap:
        push num to heap
        inHeap.add(num)
```

**⏱ Time:** O(log n) per op
**📦 Space:** O(n)

---

### **3. Maximum Subsequence Score**

**✅ Algorithm:**

* Sort (nums1\[i], nums2\[i]) by `nums2[i]` descending.
* Use **min-heap** to track top `k` values of `nums1[i]`.
* At each step, compute `sum(nums1 top k) * nums2[i]`.

**🧾 Pseudo Code:**

```pseudo
pairs = zip(nums1, nums2), sort by nums2 descending
heap = [], total = 0, maxScore = 0

for each (a, b) in pairs:
    push a to heap
    total += a
    if size > k:
        total -= pop smallest a
    if size == k:
        maxScore = max(maxScore, total * b)

return maxScore
```

**⏱ Time:** O(n log k)
**📦 Space:** O(k)

---

### **4. Total Cost to Hire K Workers**

**✅ Algorithm:**
Use two min-heaps (candidates from start and end).
Pick min cost each round and maintain size.

**🧾 Pseudo Code:**

```pseudo
leftHeap = first candidates
rightHeap = last candidates
heapify both
i, j = candidates, n - candidates - 1
totalCost = 0

while k > 0:
    choose smaller top of both heaps
    if pick left: totalCost += pop left, i += 1
    else: totalCost += pop right, j -= 1
    push next available to heap
    k -= 1

return totalCost
```

**⏱ Time:** O(k log c) where c = candidates
**📦 Space:** O(c)

Here’s a **deep explanation** of key **Binary Search** problems with:

* ✅ **Algorithm**
* 🧾 **Pseudo Code**
* ⏱ **Time Complexity**
* 📦 **Space Complexity**

---

## 🔍 Binary Search

---

### **1. Guess Number Higher or Lower**

**✅ Algorithm:**
Classic binary search from 1 to `n`.
Use `guess(num)` API:

* return -1 if high, 1 if low, 0 if correct.

**🧾 Pseudo Code:**

```pseudo
low = 1, high = n
while low <= high:
    mid = (low + high) // 2
    res = guess(mid)
    if res == 0: return mid
    elif res < 0: high = mid - 1
    else: low = mid + 1
```

**⏱ Time:** O(log n)
**📦 Space:** O(1)

---

### **2. Successful Pairs of Spells and Potions**

**✅ Algorithm:**
For each spell, use binary search on sorted potions to find count of potions that satisfy `spell * potion >= success`.

**🧾 Pseudo Code:**

```pseudo
sort potions
result = []

for spell in spells:
    left = 0, right = len(potions) - 1
    while left <= right:
        mid = (left + right) // 2
        if spell * potions[mid] >= success:
            right = mid - 1
        else:
            left = mid + 1
    result.append(len(potions) - left)
```

**⏱ Time:** O(m log n) (m = spells, n = potions)
**📦 Space:** O(m)

---

### **3. Find Peak Element**

**✅ Algorithm:**
Use binary search:

* if `nums[mid] > nums[mid + 1]`, go left (peak is mid or left).
* else go right.

**🧾 Pseudo Code:**

```pseudo
left = 0, right = n - 1
while left < right:
    mid = (left + right) // 2
    if nums[mid] > nums[mid + 1]:
        right = mid
    else:
        left = mid + 1
return left
```

**⏱ Time:** O(log n)
**📦 Space:** O(1)

---

### **4. Koko Eating Bananas**

**✅ Algorithm:**
Use binary search on eating speed `k`:

* range from 1 to max(piles).
* for each speed, calculate time needed to finish.
* find minimum `k` where total time ≤ h.

**🧾 Pseudo Code:**

```pseudo
left = 1, right = max(piles)

while left < right:
    mid = (left + right) // 2
    time = sum(ceil(pile / mid) for pile in piles)
    if time <= h:
        right = mid
    else:
        left = mid + 1
return left
```

**⏱ Time:** O(n log m) (n = piles, m = max pile size)
**📦 Space:** O(1)


Here’s a **deep dive summary** for all remaining topics you listed, categorized by:

* ✅ **Algorithm**
* 🧾 **Pseudo Code**
* ⏱ **Time Complexity**
* 📦 **Space Complexity**

---

## 🔁 Backtracking

---

### **1. Letter Combinations of a Phone Number**

**✅ Algorithm:** Recursive DFS building letter combinations using digit-to-char mapping.

**🧾 Pseudo Code:**

```pseudo
function backtrack(index, path):
    if index == len(digits): result.append(path); return
    for char in map[digits[index]]:
        backtrack(index + 1, path + char)
```

**⏱ Time:** O(4ⁿ)
**📦 Space:** O(n)

---

### **2. Combination Sum III**

**✅ Algorithm:** DFS with backtracking. Track sum and size of path.

**🧾 Pseudo Code:**

```pseudo
function backtrack(start, path, total):
    if len(path) == k and total == n: result.append(path); return
    for i from start to 9:
        if total + i > n: break
        backtrack(i + 1, path + [i], total + i)
```

**⏱ Time:** O(C(9, k))
**📦 Space:** O(k)

---

## 💠 DP – 1D

---

### **3. N-th Tribonacci Number**

**✅ Algorithm:** Bottom-up DP: T\[n] = T\[n−1] + T\[n−2] + T\[n−3]

**🧾 Pseudo Code:**

```pseudo
dp = [0, 1, 1]
for i = 3 to n:
    dp[i] = dp[i−1] + dp[i−2] + dp[i−3]
return dp[n]
```

**⏱ Time:** O(n)
**📦 Space:** O(1)

---

### **4. Min Cost Climbing Stairs**

**✅ Algorithm:** DP: cost\[i] = min(cost\[i−1], cost\[i−2]) + current

**🧾 Pseudo Code:**

```pseudo
for i = 2 to n:
    dp[i] = min(dp[i−1], dp[i−2]) + cost[i]
```

**⏱ Time:** O(n)
**📦 Space:** O(1)

---

### **5. House Robber**

**✅ Algorithm:** DP: choose to rob or skip current house

**🧾 Pseudo Code:**

```pseudo
dp[i] = max(dp[i−1], dp[i−2] + nums[i])
```

**⏱ Time:** O(n)
**📦 Space:** O(1)

---

### **6. Domino and Tromino Tiling**

**✅ Algorithm:** DP recurrence:
`dp[n] = dp[n−1] + dp[n−2] + dp[n−3]`

**🧾 Pseudo Code:**

```pseudo
dp[0] = 1, dp[1] = 1, dp[2] = 2
for i = 3 to n:
    dp[i] = dp[i−1] + dp[i−2] + dp[i−3]
```

**⏱ Time:** O(n)
**📦 Space:** O(1)

---

## 🔲 DP – Multidimensional

---

### **7. Unique Paths**

**✅ Algorithm:** Grid DP: paths\[i]\[j] = paths\[i−1]\[j] + paths\[i]\[j−1]

**🧾 Pseudo Code:**

```pseudo
for i in 1..m:
    for j in 1..n:
        dp[i][j] = dp[i−1][j] + dp[i][j−1]
```

**⏱ Time:** O(m × n)
**📦 Space:** O(n)

---

### **8. Longest Common Subsequence**

**✅ Algorithm:** Classic 2D DP

**🧾 Pseudo Code:**

```pseudo
if text1[i] == text2[j]:
    dp[i][j] = 1 + dp[i−1][j−1]
else:
    dp[i][j] = max(dp[i−1][j], dp[i][j−1])
```

**⏱ Time:** O(m × n)
**📦 Space:** O(n)

---

### **9. Best Time to Buy and Sell Stock with Transaction Fee**

**✅ Algorithm:** Track `hold` and `cash` states.

**🧾 Pseudo Code:**

```pseudo
hold = −prices[0]
cash = 0
for price in prices:
    hold = max(hold, cash − price)
    cash = max(cash, hold + price − fee)
```

**⏱ Time:** O(n)
**📦 Space:** O(1)

---

### **10. Edit Distance**

**✅ Algorithm:** Classic DP: insert, delete, replace

**🧾 Pseudo Code:**

```pseudo
if word1[i] == word2[j]:
    dp[i][j] = dp[i−1][j−1]
else:
    dp[i][j] = 1 + min(dp[i−1][j], dp[i][j−1], dp[i−1][j−1])
```

**⏱ Time:** O(m × n)
**📦 Space:** O(n)

---

## ⚙️ Bit Manipulation

---

### **11. Counting Bits**

**✅ Algorithm:**
`dp[i] = dp[i >> 1] + (i & 1)`

**🧾 Pseudo Code:**

```pseudo
for i = 1 to n:
    dp[i] = dp[i >> 1] + (i & 1)
```

**⏱ Time:** O(n)
**📦 Space:** O(n)

---

### **12. Single Number**

**✅ Algorithm:** XOR all numbers

**🧾 Pseudo Code:**

```pseudo
res = 0
for num in nums:
    res ^= num
return res
```

**⏱ Time:** O(n)
**📦 Space:** O(1)

---

### **13. Minimum Flips to Make a OR b Equal to c**

**✅ Algorithm:** Bitwise comparison on each bit of a, b, c.

**🧾 Pseudo Code:**

```pseudo
for i in 0..31:
    bitA = a >> i & 1
    bitB = b >> i & 1
    bitC = c >> i & 1
    if (bitA | bitB) != bitC:
        if bitC == 1: flips += 1
        else: flips += bitA + bitB
```

**⏱ Time:** O(1)
**📦 Space:** O(1)

---

## 🔡 Trie

---

### **14. Implement Trie (Prefix Tree)**

**✅ Algorithm:** Classic Trie Node with insert, search, startsWith

**🧾 Pseudo Code:**

```pseudo
class TrieNode:
    children = map
    isEnd = False

insert(word): traverse chars, create nodes
search(word): traverse, check isEnd
startsWith(prefix): traverse, check path exists
```

**⏱ Time:** O(m) per operation
**📦 Space:** O(m × n)

---

### **15. Search Suggestions System**

**✅ Algorithm:** Build trie, use DFS for prefix search (limit to 3 suggestions)

**🧾 Pseudo Code:**

```pseudo
build trie from products
for char in searchWord:
    move to next node
    collect top 3 suggestions using DFS
```

**⏱ Time:** O(n log n + m²)
**📦 Space:** O(n)

---

## 🧭 Intervals

---

### **16. Non-overlapping Intervals**

**✅ Algorithm:**
Sort by end time, use greedy to skip overlapping intervals.

**🧾 Pseudo Code:**

```pseudo
sort intervals by end
prevEnd = −∞
for interval in intervals:
    if interval.start >= prevEnd:
        prevEnd = interval.end
    else:
        remove++
```

**⏱ Time:** O(n log n)
**📦 Space:** O(1)

---

### **17. Minimum Number of Arrows to Burst Balloons**

**✅ Algorithm:**
Greedy – sort by end time, shoot arrows at end of current overlapping interval.

**🧾 Pseudo Code:**

```pseudo
sort balloons by end
arrows = 1, prevEnd = balloons[0].end
for each balloon:
    if balloon.start > prevEnd:
        arrows += 1
        prevEnd = balloon.end
```

**⏱ Time:** O(n log n)
**📦 Space:** O(1)

---

## 📈 Monotonic Stack

---

### **18. Daily Temperatures**

**✅ Algorithm:**
Monotonic stack to track indices of decreasing temps.

**🧾 Pseudo Code:**

```pseudo
for i = n-1 to 0:
    while stack not empty and temp[stack.top()] <= temp[i]:
        stack.pop()
    result[i] = stack.top() − i if stack else 0
    stack.push(i)
```

**⏱ Time:** O(n)
**📦 Space:** O(n)

---

### **19. Online Stock Span**

**✅ Algorithm:**
Monotonic stack holding (price, span)

**🧾 Pseudo Code:**

```pseudo
while stack not empty and price >= stack.top().price:
    span += stack.pop().span
stack.push({price, span})
return span
```

**⏱ Time:** Amortized O(1) per call
**📦 Space:** O(n)






