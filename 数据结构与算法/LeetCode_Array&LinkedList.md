## LeetCode 数组&链表题解 

### [LeetCode 283. 移动零](https://leetcode-cn.com/problems/move-zeroes/)

给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。

```java
class Solution {
    public void moveZeroes(int[] nums) {
        if (nums == null || nums.length == 0)
            return;
        int j = 0;
        for (int num : nums) {
            if (num != 0) {
                nums[j++] = num;
            }
        }
        while (j < nums.length) {
            nums[j++] = 0;
        }
    }
}
```

### [LeetCode 11. 盛最多水的容器](https://leetcode-cn.com/problems/container-with-most-water/)

![](https://aliyun-lc-upload.oss-cn-hangzhou.aliyuncs.com/aliyun-lc-upload/uploads/2018/07/25/question_11.jpg)

```java
/**
 * 最优解 ==> 使用双指针
 */
class Solution {
    public int maxArea(int[] height) {
        int left = 0;
        int right = height.length - 1;
        int maxArea = 0;
        while (left < right) {
            maxArea = Math.max(maxArea, Math.min(height[left], height[right]) * (right - left));
            if (height[left] < height[right]) {
                left++;
            } else {
                right--;
            }
        }
        return maxArea;
    }
}
```

### [LeetCode 42. 接雨水](https://leetcode-cn.com/problems/trapping-rain-water/) (★★★)

给定 *n* 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/10/22/rainwatertrap.png)

```java

```

### [LeetCode2. 两数相加 ](https://leetcode-cn.com/problems/add-two-numbers/)(★★★)

```java
/**
 * 两个链表各自的位数是按照逆序的方式存储的
 * 如果需要正序输出可以采用反转单链表的方式
 */
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        int carry = 0;
        ListNode dummy = new ListNode(0);
        ListNode prev = dummy;
        while (l1 != null || l2 != null || carry != 0) {
            int x = l1 == null ? 0 : l1.val;
            int y = l2 == null ? 0 : l2.val;
            int sum = x + y + carry;
            carry = sum / 10;
            ListNode node = new ListNode(sum % 10);
            prev.next = node;
            prev = node;
            if (l1 != null) l1 = l1.next;
            if (l2 != null) l2 = l2.next;
        }
        return dummy.next;
    }
}
```

### [LeetCode 15. 三数之和 ](https://leetcode-cn.com/problems/3sum/)(★★★★)

给定一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？找出所有满足条件且不重复的三元组。

```java

```

