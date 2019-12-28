## 剑指offer题解

### 数组中重复的数字

在一个长度为 n 的数组里的所有数字都在 0 到 n-1 的范围内。数组中某些数字是重复的，但不知道有几个数字是重复的，也不知道每个数字重复几次。请找出数组中任意一个重复的数字。    

```java
/**
* 对于这种数组元素在 [0, n-1] 范围内的问题，可以将值为 i 的元素调整到第 i 个位置上进行求解
* 时间复杂度 O(N)，空间复杂度 O(1)
*/
public class Solution {
    public boolean duplicate(int numbers[], int length, int[] duplication) {
        if (numbers == null || length == 0)
            return false;
        for (int i = 0; i < length; i++) {
            while (numbers[i] != i) {
                if (numbers[i] == numbers[numbers[i]]) {
                    duplication[0] = numbers[i];
                    return true;
                }
                swap(numbers, i, numbers[i]);
            }
        }
        return false;
    }
 
    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}
```

### 二维数组中的查找    

给定一个二维数组，其每一行从左到右递增排序，从上到下也是递增排序。给定一个数，判断这个数是否在该二维数 组中。    

```java
/**
 * 从右上角开始查找，就可以根据 target 和当前元素的大小关系来缩小查找区间
 * 时间复杂度 O(M + N)，空间复杂度 O(1)
 */
public class Solution {
    public boolean Find(int target, int [][] array) {
        if (array == null || array[0].length == 0){
            return false;
        }
        int rows = array.length;
        int cols = array[0].length;
        int r = 0;
        int c = cols - 1;
        while (r < rows && c >= 0){
            if (target > array[r][c]){
                r++;
            }else if (target < array[r][c]) {
                c--;
            }else {
                return true;
            }
        }
        return false;
    }
}
```

### 替换空格 

将一个字符串中的空格替换成 "%20"    

   ```java
/**
 * 在字符串尾部填充任意字符，使得字符串的长度等于替换之后的长度
 * 令 p1 指向字符串原来的末尾位置，p2 指向字符串现在的末尾位置
 */
public class Solution {
    public String replaceSpace(StringBuffer str) {
        int p1 = str.length() - 1;
        for (int i = 0;i <= p1;i++){
            if (str.charAt(i) == ' '){
                str.append("  ");
            }
        }
        int p2 = str.length() - 1;
        while (p1 >= 0 && p2 >= p1){
            char c = str.charAt(p1--);
            if (c == ' '){
                str.setCharAt(p2--,'0');
                str.setCharAt(p2--,'2');
                str.setCharAt(p2--,'%');
            }else {
                str.setCharAt(p2--,c);
            }
        }
        return str.toString();
    }
}
   ```

### 从尾到头打印链表 

```java
/**
 * 从尾到头反过来打印出链表每个结点的值
 * 借助栈 ==> 不改变链表的结构
 */
public class Solution {
    public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
        Stack<ListNode> stack = new Stack<>();
        ArrayList<Integer> list = new ArrayList<>();
        while (listNode != null) {
            stack.push(listNode);
            listNode = listNode.next;
        }

        while (!stack.isEmpty()) {
            list.add(stack.pop().val);
        }
        return list;
    }
}
```

### 重建二叉树

根据二叉树的前序遍历和中序遍历的结果，重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。    

### 斐波那契数列

```java
public class Solution {
    public int Fibonacci(int n) {
        if (n <= 1)
            return n;
        int[] dp = new int[n + 1];
        dp[1] = 1;
        for (int i = 2;i <= n;i++){
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n];
    }
}
```

#### 跳台阶  

```java
/**
 *  dp[i] = dp[i - 1] + dp[i - 2] 类问题
 *  考虑到 dp[i] 只与 dp[i - 1] 和 dp[i - 2] 有关
 *  因此可以只用两个变量来存储 dp[i - 1] 和 dp[i - 2]
 *  使得原来的 O(N)空间复杂度优化为 O(1) 复杂度
 */
public class Solution {
    public int JumpFloor(int target) {
        if (target <= 2)
            return target;
        int prev1 = 1;
        int prev2 = 2;
        int curr = 0;
        for (int i = 3;i <= target;i++){
            curr = prev1 + prev2;
            prev1 = prev2;
            prev2 = curr;
        }
        return curr;
    }
}
```

### 旋转数组的最小数字

把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。输入一个非递减排序的数组的一个旋转，输出旋转数组的最小元素。

```java
/**
 * 将旋转数组对半分可以得到一个包含最小元素的新旋转数组，以及一个非递减排序的数组
 * 新的旋转数组的数组元素是原数组的一半，从而将问题规模减少了一半
 * 这种折半性质的算法的时间复杂度为 O(logN)
 * 如果数组元素允许重复，会出现一个特殊的情况：nums[l] == nums[m] == nums[h]
 * 例如 数组 {1,1,1,0,1}
 */
public class Solution {
    public int minNumberInRotateArray(int[] array) {
        if (array.length == 0 || array == null)
            return 0;
        int l = 0;
        int h = array.length - 1;
        while (l < h) {
            int m = l + (h - l) / 2;
            if (array[m] == array[l] && array[m] == array[h])
                return minNumber(array, l, h);
            else if (array[m] <= array[h])
                h = m;
            else
                l = m + 1;
        }
        return array[l];
    }

    private int minNumber(int[] nums, int l, int h) {
        for (int i = l; i < h; i++) {
            if (nums[i] > nums[i + 1]) {
                return nums[i + 1];
            }
        }
        return nums[l];
    }
}
```

### 矩阵中的路径

判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一个格子开始，每一步可以 在矩阵中向上下左右移动一个格子。如果一条路径经过了矩阵中的某一个格子，则该路径不能再进入该格子。    

```java
/**
 * 使用回溯法（backtracking）进行求解，它是一种暴力搜索方法，通过搜索所有可能的结果来求解问题。
 * 回溯法在一次搜索结束时需要进行回溯（回退），将这一次搜索过程中设置的状态进行清除，
 * 从而开始一次新的搜索过程
 */
public class Solution {

    private final static int[][] next = {{0, -1}, {0, 1}, {-1, 0}, {1, 0}};
    private int rows;
    private int cols;

    public boolean hasPath(char[] array, int rows, int cols, char[] str) {
        if (rows == 0 || cols == 0) return false;
        this.rows = rows;
        this.cols = cols;
        boolean[][] marked = new boolean[rows][cols];
        char[][] matrix = buildMatrix(array);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (backtracking(matrix, str, marked, 0, i, j))
                    return true;
            }
        }
        return false;
    }

    private boolean backtracking(char[][] matrix, char[] str,boolean[][] marked, int pathLen, int r, int c) {
        if (pathLen == str.length) return true;
        if (r < 0 || r >= rows || c < 0 || c >= cols
                || matrix[r][c] != str[pathLen] || marked[r][c]) {
            return false;
        }

        marked[r][c] = true;
        for (int[] n : next) {
            if (backtracking(matrix, str, marked, pathLen + 1, r + n[0], c + n[1]))
                return true;
        }
        marked[r][c] = false;
        return false;
    }

    private char[][] buildMatrix(char[] matrix){
        char[][] newMatrix = new char[rows][cols];
        int p = 0;
        for (int i = 0;i < rows;i++){
            for (int j = 0;j < cols;j++){
                newMatrix[i][j] = matrix[p++];
            }
        }
        return newMatrix;
    }
}
```

```java
public class Solution {
    public boolean hasPath(char[] array, int rows, int cols, char[] str) {
        if (rows == 0 || cols == 0) return false;
        boolean[] marked = new boolean[array.length];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (backtracking(array, str, marked, 0, rows, cols, i, j))
                    return true;
            }
        }
        return false;
    }

    private boolean backtracking(char[] matrix, char[] str, boolean[] marked, int pathLen, int rows, int cols, int i, int j) {
        if (pathLen == str.length) return true;
        int index = i * cols + j;
        if (i < 0 || i >= rows || j < 0 || j >= cols
                || matrix[index] != str[pathLen] || marked[index]) {
            return false;
        }
        marked[index] = true;
        if (backtracking(matrix, str, marked, pathLen + 1, rows, cols, i + 1, j) ||
                backtracking(matrix, str, marked, pathLen + 1, rows, cols, i - 1, j) ||
                backtracking(matrix, str, marked, pathLen + 1, rows, cols, i, j + 1) ||
                backtracking(matrix, str, marked, pathLen + 1, rows, cols, i, j - 1))
            return true;
        marked[index] = false;
        return false;
    }
}
```

### 机器人的运动范围 

地上有一个 m 行和 n 列的方格。一个机器人从坐标 (0, 0) 的格子开始移动，每一次只能向左右上下四个方向移动一 格，但是不能进入行坐标和列坐标的数位之和大于 k 的格子 

```java
public class Solution {
    private static final int[][] next = {{0, -1}, {0, 1}, {-1, 0}, {1, 0}};
    private int cnt = 0;
    private int rows;
    private int cols;
    private int threshold;
    private int[][] digitSum;

    public int movingCount(int threshold, int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.threshold = threshold;
        initDigitSum();
        boolean[][] marked = new boolean[rows][cols];
        DFS(marked, 0, 0);
        return cnt;
    }

    private void DFS(boolean[][] marked, int i, int j) {
        if (i < 0 || i >= rows || j < 0 || j >= cols || marked[i][j] || threshold < digitSum[i][j])
            return;
        marked[i][j] = true;
        cnt++;
        for (int[] n : next) {
            DFS(marked, n[0] + i, n[1] + j);
        }
    }

    private void initDigitSum() {
        int[] digitSumOne = new int[Math.max(rows, cols)];

        for (int i = 0; i < digitSumOne.length; i++) {
            int n = i;
            while (n > 0) {
                digitSumOne[i] += n % 10;
                n = n / 10;
            }
        }
        this.digitSum = new int[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                digitSum[i][j] = digitSumOne[i] + digitSumOne[j];
            }
        }
    }
}
```

### 剪绳子

把一根绳子剪成多段，并且使得每段的长度乘积最大

```java
/**
 * 贪心思想
 * 尽可能多剪长度为 3 的绳子，并且不允许有长度为 1 的绳子出现。
 * 如果出现了，就从已经切好长度为 3 的绳子中拿出一段与长度为 1 的绳子重新组合，
 * 把它们切成两段长度为 2 的绳子。
 */
public class Solution {
    public int cutRope(int n) {
        if (n < 2)
            return 0;
        if (n == 2)
            return 1;
        if (n == 3)
            return 2;
        int timesOfThree = n /3;
        if ((n - timesOfThree * 3) == 1){
            timesOfThree--;
        }
        int timesOfTwo = (n - timesOfThree*3)/2;

        return (int) Math.pow(3,timesOfThree) * (int) Math.pow(2,timesOfTwo);
    }
}
```

### 二进制中 1 的个数    

```java
/**
 * 位运算
 * n&(n-1) 去除 n 的位级表示中最低的那一位
 */
public class Solution {
    public int NumberOf1(int n) {
        int count = 0;
        while (n != 0){
            n &= n - 1;
            count++;
        }
        return count;
    }
}
```

### 数组中不重复的两个元素 

一个整型数组里除了两个数字之外，其他的数字都出现了两次

```java
/**
 * 两个不相等的元素在位级表示上必定会有一位存在不同
 * 将数组的所有元素异或得到的结果为不存在重复的两个元素异或的结果
 * diff &= -diff 得到出 diff 最右侧不为 0 的位
 */
public class Solution {
    public void FindNumsAppearOnce(int [] nums,int num1[] , int num2[]) {
        int diff = 0;
        for (int num : nums){
            diff ^= num;
        }
        diff &= -diff;
        for (int num : nums){
            if ((num & diff) == 0) {
                num1[0] ^= num;
            } else {
                num2[0] ^= num;
            }
        }
    }
}
```

### 数值的整数次方

给定一个 double 类型的浮点数 base 和 int 类型的整数 exponent，求 base 的 exponent 次方

```java
/**
 * 使用递归求解，时间复杂度 O(n)
 * 代码的完整性 ==> 考虑到 exponent 为负的情况
 */
public class Solution {
    public double Power(double base, int exponent) {
        if (exponent == 0)
            return 1;
        if (exponent == 1)
            return base;
        boolean flag = false;
        if (exponent < 0) {
            exponent = -exponent;
            flag = true;
        }

        double power = Power(base * base, exponent / 2);
        if (exponent % 2 != 0) {
            power = power * base;
        }

        return flag ? 1 / power : power;
    }
}
```

### 打印从 1 到最大的 n 位数

输入数字 n，按顺序打印出从 1 到最大的 n 位十进制数。比如输入 3，则打印出 1、2、3 一直到最大的 3 位数即 999

### 删除链表中重复的结点

```java
/**
 * 构建哑结点 dummy, 不用考虑重复的结点是否从head开始
 * 不保留重复的结点
 */
public class Solution {
    public ListNode deleteDuplication(ListNode head) {
        if (head == null || head.next == null)
            return head;
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode prev = dummy;
        ListNode curr = head;
        while (curr != null && curr.next != null) {
            if (curr.val == curr.next.val) {
                while (curr.next != null && curr.val == curr.next.val) {
                    curr = curr.next;
                }
                prev.next = curr.next;
                curr = curr.next;
            } else {
                prev = curr;
                curr = curr.next;
            }
        }
        return dummy.next;
    }
}
```

#### [LeetCode 83. 删除排序链表中的重复元素](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list/)

```java
/**
 * 保留重复的结点
 * 可以不构建哑结点
 */
public class Solution {
    public ListNode deleteDuplication(ListNode head) {
        if (head == null || head.next == null)
            return head;
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode prev = dummy;
        ListNode curr = head;
        while (curr != null && curr.next != null) {
            if (curr.val == curr.next.val) {
                while (curr.next != null && curr.val == curr.next.val) {
                    curr = curr.next;
                }
                prev.next = curr;
                prev = curr;
                curr = curr.next;
            } else {
                prev = curr;
                curr = curr.next;
            }
        }
        return dummy.next;
    }
}
```

### 正则表达式匹配

请实现一个函数用来匹配包括 ' . ' 和 ' * ' 的正则表达式。模式中的字符 ' . ' 表示任意一个字符，而 ' * ' 表示它前面的字符可以出现任意次（包含 0 次）。

```java
public class Solution {
    public boolean match(char[] str, char[] pattern) {
        if (str == null || pattern == null) {
            return false;
        }
        return matchCore(str, 0, pattern, 0);
    }
    private boolean matchCore(char[] str, int strIndex, char[] pattern, int patternIndex) {
        // 有效性检测
        if (strIndex == str.length && patternIndex == pattern.length) {
            return true;
        }
        // pattern先到尾匹配失败
        if (strIndex != str.length && patternIndex == pattern.length) {
            return false;
        }
        // 第一个字符匹配且 pattern 第二个字符为* , 分三种匹配情况
        // 如果不匹配, 模式后移2位
        if ((patternIndex + 1) < pattern.length && pattern[patternIndex + 1] == '*') {
            if ((strIndex < str.length && str[strIndex] == pattern[patternIndex]) ||
                    (strIndex < str.length && pattern[patternIndex] == '.')) {
               return matchCore(str, strIndex, pattern, patternIndex + 2) || // 视为x*匹配0个字符
                        matchCore(str, strIndex + 1, pattern, patternIndex + 2) ||
                        	matchCore(str, strIndex + 1, pattern, patternIndex);
            } else {
                return matchCore(str, strIndex, pattern, patternIndex + 2);
            }
        }
        //模式第2个不是*，且字符串第1个跟模式第1个匹配，则都后移1位，否则直接返回false
        if ((strIndex != str.length && pattern[patternIndex] == str[strIndex]) || 
                (pattern[patternIndex] == '.' && strIndex != str.length)) {
            return matchCore(str, strIndex + 1, pattern, patternIndex + 1);
        }
        return false;
    }
}
```

#### 表示数值的字符串

```java
/**
 *  []  字符集合
 *  ()  分组
 *  ?   重复 0 ~ 1 次
 *  +   重复 1 ~ n 次
 *  *   重复 0 ~ n 次
 *  .   任意字符
 *  \\. 转义后的 .
 *  \\d 数字
 */
public class Solution {
    public boolean isNumeric(char[] str) {
        if (str == null || str.length == 0)
            return false;
        String string = new String(str);
        return string.matches("[+-]?\\d*(\\.\\d+)?([Ee][+-]?\\d+)?");
    }
}
```

### 调整数组顺序使奇数位于偶数前面

```java
/**
 * 创建一个新数组，时间复杂度 O(N)，空间复杂度 O(N)
 */
public class Solution {
    public void reOrderArray(int[] nums) {
        int lenOfOdd = 0;
        for (int num : nums) {
            if (num % 2 != 0)
                lenOfOdd++;
        }
        int[] copyNums = nums.clone();
        int i = 0;
        for (int num : copyNums) {
            if (num % 2 != 0) {
                nums[i++] = num;
            } else {
                nums[lenOfOdd++] = num;
            }
        }
    }
}
```

### 返回链表中倒数第 K 个结点

```java
/**
 * 使用双指针
 */
public class Solution {
    public ListNode FindKthToTail(ListNode head,int k) {
        if (head == null){
            return null;
        }
        ListNode p1 = head;
        ListNode p2 = head;
        while (k-- > 0){
            if (p1 == null){
                return null;
            }
            p1 = p1.next;
        }
        while (p1 != null){
            p1 = p1.next;
            p2 = p2.next;
        }
        return p2;
    }
}
```

#### [LeetCode 19. 删除链表的倒数第N个节点](https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/)

```java
/**
 * 使用双指针
 * 使用哑结点 dummy, 考虑删除的结点是头结点
 */
class Solution {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        if (head == null){
            return null;
        }
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode p1 = dummy;
        ListNode p2 = dummy;
        while (n-- > 0){
            p1 = p1.next;
        }
        while (p1.next != null){
            p1 = p1.next;
            p2 = p2.next;
        }
        p2.next = p2.next.next;
        return dummy.next;
    }
}
```

### 链表中环的入口结点

```java
public class Solution {
    public ListNode EntryNodeOfLoop(ListNode head) {
        if (head == null || head.next == null)
            return null;
        ListNode slow = head;
        ListNode fast = head;
        do {
            if (fast == null || fast.next == null)
                return null;
            slow = slow.next;
            fast = fast.next.next;
        }while (slow != fast);
        slow = head;
        while (slow != fast){
            slow = slow.next;
            fast = fast.next;
        }
        return slow;
    }
}
```

### 反转链表

```java
/**
 * 递归
 */
public class Solution {
    public ListNode ReverseList(ListNode head) {
       if (head == null || head.next == null)
           return head;
       ListNode temp = head.next;
       ListNode newHead = ReverseList(temp);
       temp.next = head;
       head.next = null;
       return newHead;
    }
}
/**
 * 迭代
 */
public class Solution {
    public ListNode ReverseList(ListNode head) {
       if (head == null || head.next == null)
           return head;
       ListNode prev = null;
       ListNode curr = head;
       while (curr != null){
           ListNode temp = curr.next;
           curr.next = prev;
           prev = curr;
           curr = temp;
       }
       return prev;
    }
}
```

### 合并两个排序的链表

```java
public class Solution {
    public ListNode Merge(ListNode list1,ListNode list2) {
        ListNode dummy = new ListNode(0);
        ListNode p = dummy;
        ListNode p1 = list1;
        ListNode p2 = list2;
        while (p1 != null && p2 != null){
            if (p1.val <= p2.val){
                p.next = p1;
                p = p1;
                p1 = p1.next;
            }else {
                p.next = p2;
                p = p2;
                p2 = p2.next;
            }
        }
        p.next = p1 == null ? p2 : p1;
        return dummy.next;
    }
}
```

#### [LeetCode 23. 合并K个排序链表](https://leetcode-cn.com/problems/merge-k-sorted-lists/)

```java
public class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        return mergeKLists(lists,0,lists.length - 1);
    }

    private ListNode mergeKLists(ListNode[] lists, int left, int right) {
        if (left == right)
            return lists[left];
        int mid = (left + right) / 2;
        ListNode l1 = mergeKLists(lists, left, mid);
        ListNode l2 = mergeKLists(lists, mid + 1, right);
        return mergeTwoLists(l1, l2);
    }

    private ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        ListNode dummy = new ListNode(0);
        ListNode p = dummy;
        ListNode p1 = list1;
        ListNode p2 = list2;
        while (p1 != null && p2 != null) {
            if (p1.val <= p2.val) {
                p.next = p1;
                p = p1;
                p1 = p1.next;
            } else {
                p.next = p2;
                p = p2;
                p2 = p2.next;
            }
        }
        p.next = p1 == null ? p2 : p1;
        return dummy.next;
    }
}
```

### 树的子结构

```java
public class Solution {
    public boolean HasSubtree(TreeNode root1,TreeNode root2) {
        if (root1 == null || root2 == null)
            return false;
        return isSubtreeWithRoot(root1,root2) ||
                    HasSubtree(root1.left,root2) ||
                        HasSubtree(root1.right,root2);
    }
    private boolean isSubtreeWithRoot(TreeNode root1, TreeNode root2){
        if (root2 == null)
            return true;
        if (root1 == null)
            return false;
        if (root1.val != root2.val)
            return false;
        return isSubtreeWithRoot(root1.left,root2.left) &&
                    isSubtreeWithRoot(root1.right,root2.right);
    }
}
```

#### [LeetCode 572. 另一个树的子树](https://leetcode-cn.com/problems/subtree-of-another-tree/)

```java
class Solution {
    public boolean isSubtree(TreeNode s, TreeNode t) {
        if (s == null)
            return false;
        return isSubtreeWithRoot(s,t) || isSubtree(s.left, t) || isSubtree(s.right,t);
    }
    private boolean isSubtreeWithRoot(TreeNode s, TreeNode t){
        if (s == null && t == null)
            return true;
        if (s == null || t == null)
            return false;
        if (s.val != t.val)
            return false;
        return isSubtreeWithRoot(s.left,t.left) && isSubtreeWithRoot(s.right,t.right);
    }
}
```

### 二叉树的镜像

```java
public class Solution {
    public void Mirror(TreeNode root) {
        root = invertTree(root);
    }
    public TreeNode invertTree(TreeNode root) {
        if (root == null)
            return null;
        TreeNode left = root.left;
        root.left = invertTree(root.right);
        root.right = invertTree(left);
        return root;
    }
}
```

#### [LeetCode 226. 翻转二叉树](https://leetcode-cn.com/problems/invert-binary-tree/)

```java
class Solution {
    public TreeNode invertTree(TreeNode root) {
        if (root == null)
            return null;
        TreeNode left = root.left;
        root.left = invertTree(root.right);
        root.right = invertTree(left);
        return root;
    }
}
```

#### [LeetCode 101. 对称二叉树](https://leetcode-cn.com/problems/symmetric-tree/)

```java
class Solution {
    public boolean isSymmetric(TreeNode root) {
        if (root == null)
            return true;
        return isSymmetric(root.left, root.right);
    }

    private boolean isSymmetric(TreeNode root1, TreeNode root2) {
        if (root1 == null && root2 == null)
            return true;
        if (root1 == null || root2 == null)
            return false;
        if (root1.val != root2.val)
            return false;
        return isSymmetric(root1.left, root2.right) && isSymmetric(root1.right, root2.left);
    }
}
```

### 顺时针打印矩阵

```java
public class Solution {
    public ArrayList<Integer> printMatrix(int[][] matrix) {
        ArrayList<Integer> list = new ArrayList<>();
        int r1 = 0, r2 = matrix.length - 1;
        int c1 = 0, c2 = matrix[0].length - 1;
        while (r1 <= r2 && c1 <= c2) {
            for (int i = c1; i <= c2; i++) {
                list.add(matrix[r1][i]);
            }
            for (int i = r1 + 1; i <= r2; i++) {
                list.add(matrix[i][c2]);
            }
            if (r1 != r2)
                for (int i = c2 - 1; i >= c1; i--)
                    list.add(matrix[r2][i]);
            if (c1 != c2)
                for (int i = r2 - 1; i > r1; i--)
                    list.add(matrix[i][r1]);
            r1++;
            r2--;
            c1++;
            c2--;
        }
        return list;
    }
}
```

### 从上往下打印二叉树(二叉树的层次遍历)

```java
public class Solution {
    public ArrayList<Integer> PrintFromTopToBottom(TreeNode root) {
        ArrayList<Integer> list = new ArrayList<>();
        if (root == null)
            return list;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            list.add(node.val);
            if (node.left != null)
                queue.add(node.left);
            if (node.right != null)
                queue.add(node.right);
        }
        return list;
    }
}
```

#### [LeetCode 102. 二叉树的层次遍历](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/)(把二叉树打印成多行)

```java
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> lists = new ArrayList<>();
        if (root == null)
            return lists;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            List<Integer> list = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                TreeNode node = queue.poll();
                list.add(node.val);
                if (node.left != null)
                    queue.add(node.left);
                if (node.right != null)
                    queue.add(node.right);
            }
            lists.add(list);
        }
        return lists;
    }
}
```

#### [LeetCode 103. 二叉树的锯齿形层次遍历](https://leetcode-cn.com/problems/binary-tree-zigzag-level-order-traversal/)(二叉树的之字型遍历)

```java
class Solution {
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> lists = new ArrayList<>();
        if (root == null)
            return lists;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        boolean flag = false;
        while (!queue.isEmpty()) {
            int size = queue.size();
            List<Integer> list = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                TreeNode node = queue.poll();
                list.add(node.val);
                if (node.left != null)
                    queue.add(node.left);
                if (node.right != null)
                    queue.add(node.right);
            }
            if (flag)
                Collections.reverse(list);
            flag = ! flag;
            lists.add(list);
        }
        return lists;
    }
}
```

#### [LeetCode 637. 二叉树的层平均值](https://leetcode-cn.com/problems/average-of-levels-in-binary-tree/)

```java
class Solution {
    public List<Double> averageOfLevels(TreeNode root) {
        List<Double> list = new ArrayList<>();
        if (root == null)
            return list;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            double sum = 0;
            for (int i = 0; i < size; i++) {
                TreeNode node = queue.poll();
                sum += node.val;
                if (node.left != null)
                    queue.add(node.left);
                if (node.right != null)
                    queue.add(node.right);
            }
            list.add(sum / size);
        }
        return list;
    }
}
```

### 二叉树中和为某一值的路径

输入一颗二叉树和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。路径定义为从树的根结点开始往下 一直到叶结点所经过的结点形成一条路径。

```java
private ArrayList<ArrayList<Integer>> lists = new ArrayList<>();

    public ArrayList<ArrayList<Integer>> FindPath(TreeNode root, int target) {
        backtracking(root, target, new ArrayList<>());
        return lists;
    }

    public void backtracking(TreeNode root, int target, ArrayList<Integer> list) {
        if (root == null)
            return;
        list.add(root.val);
        target -= root.val;
        if (root.left == null && root.right == null && target == 0) {
            // add添加的是引用，如果不new一个的话，后面的操作会更改这个list
            lists.add(new ArrayList<>(list));
        } else {
            backtracking(root.left, target, list);
            backtracking(root.right, target, list);
        }
        // 移除最后一个元素，深度遍历完一条路径要回退
        list.remove(list.size() - 1);
    }
```

#### [LeetCode 112. 路径总和](https://leetcode-cn.com/problems/path-sum/)

```java
class Solution {
    public boolean hasPathSum(TreeNode root, int sum) {
        if (root == null)
            return false;
        sum -= root.val;
        if (root.left == null && root.right == null && sum == 0)
            return true;
        return hasPathSum(root.left, sum) || hasPathSum(root.right, sum);
    }
}
```

#### [LeetCode 113. 路径总和 II](https://leetcode-cn.com/problems/path-sum-ii/)

```java
class Solution {
    private List<List<Integer>> lists = new ArrayList<>();
    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        backtracking(root, sum, new ArrayList<>());
        return lists;
    }

    private void backtracking(TreeNode root, int sum, List<Integer> list) {
        if (root == null)
            return;
        list.add(root.val);
        sum -= root.val;
        if (root.left == null & root.right == null && sum == 0) {
            lists.add(new ArrayList<>(list));
        } else {
            backtracking(root.left, sum, list);
            backtracking(root.right, sum, list);
        }
        list.remove(list.size() - 1);
    }
}
```

#### [LeetCode 437. 路径总和 III](https://leetcode-cn.com/problems/path-sum-iii/)

```java
class Solution {
    private int res = 0;
    public int pathSum(TreeNode root, int sum) {
        pathSumWithRoot(root, sum);
        if (root != null) {
            pathSum(root.left, sum);
            pathSum(root.right, sum);
        }
        return res;
    }

    private void pathSumWithRoot(TreeNode root, int sum) {
        if (root == null)
            return;
        if (sum == root.val) {
            res++;
        }
        // 注意这里不要使用else, 有 {1,-2} {1,-2,1,-1}的情况
        pathSumWithRoot(root.left, sum - root.val);
        pathSumWithRoot(root.right, sum - root.val);
    }
}
```

### 数组中出现次数超过一半的数字

```java
/**
 * Boyer-Moore 投票算法
 */
public class Solution {
    public int MoreThanHalfNum_Solution(int [] array) {
        int candidate = array[0];
        for(int i = 1,count = 1;i < array.length;i++){
            count += (array[i] == candidate) ? 1 : - 1;
            if (count == 0){
                candidate = array[i];
                count = 1;
            }
        }
        int cnt = 0;
        for (int num : array){
            if (num == candidate)
                cnt++;
        }
        if (cnt > array.length /2)
            return candidate;
        return 0;
    }
}
```

#### [LeetCode 169. 多数元素](https://leetcode-cn.com/problems/majority-element/)

```java
class Solution {
    public int majorityElement(int[] nums) {
        int count = 0;
        int candidate = 0;
        for (int num : nums){
            if (count == 0){
                candidate = num;
            }
            count += (num == candidate) ? 1 : -1;
        }
        return candidate;
    }
}
```

### 最小的 K 个数

