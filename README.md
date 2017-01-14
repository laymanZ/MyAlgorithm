# MyAlgorithm
## 记录一些算法题目的解法 (Record some algorithms to solve the problem)

### 1、二维数组中的查找 

题目描述:
在一个二维数组中，每一行都按照从左到右递增的顺序排序，
每一列都按照从上到下递增的顺序排序。请完成一个函数，
输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

```Java
public class Num1 
{
	  public boolean Find(int target, int [][] array) 
	    {
			int col = 0;
	        int row = array.length - 1;
	        while(col<array[0].length && row>=0)
	        {
	            if(array[row][col] == target)
	            {
	                return true;
	            }
	            else if(array[row][col] < target)
	            {
	                col++;
	            }
	            else
	            {
	                row--;
	            }
	        }
	        return false;
	    }
}

```

### 2、替换空格 
题目描述：请实现一个函数，将一个字符串中的空格替换成“%20”。
例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。

```Java
public class StringReplaceSpace 
{
	public String replaceSpace(StringBuffer str)
    {
    	String s = str.toString().replace(" ","%20");
        return s;
    }
}
```

### 3、从尾到头打印链表 
题目描述：输入一个链表，从尾到头打印链表每个节点的值
```Java
public class PrintListNum 
{

	public class ListNode
	{
	      int val;
	      ListNode next = null;
	      ListNode(int val) 
	      {
	          this.val = val;
	      }
	 }
	
	//需要注意的地方，当 listNode 为null的时候，不能返回null，要返回一个 new ArrayList， 因为  new ArrayList（） == null  的结果为false
	public ArrayList<Integer> printListFromTailToHead(ListNode listNode) 
	 {
	        if(listNode == null)
	            return new ArrayList();
	        Stack<Integer> stack = new Stack<Integer>();
	        while(listNode != null)
	        {
	            stack.push(listNode.val);
	            listNode = listNode.next;
	        }
	        ArrayList<Integer> arr = new ArrayList<Integer>();
	        while(!stack.isEmpty())
	        {
	            arr.add(stack.pop());
	        }
	        return arr;
	        
	 } 
}
```

### 4、重建二叉树 
题目描述：输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。

假设输入的前序遍历和中序遍历的结果中都不含重复的数字。

例如输入前序遍历序列{1,2,4,7,3,5,6,8}

和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。

```Java
public class ReConstructBinaryTree 
{
	 public class TreeNode 
	 {
	     int val;
	     TreeNode left;
	     TreeNode right;
	     TreeNode(int x)
       { 
        val = x; 
       }
	 }
	    public TreeNode reConstructBinaryTree(int [] pre,int [] in) 
      {
	        int i=0;
	        if(pre.length!=in.length||pre.length==0||in.length==0)
	            return null;
	        TreeNode root = new TreeNode(pre[0]);
	        while(in[i]!=root.val)
	            i++;
	        int[] preLeft = new int[i];
	        int[] inLeft = new int[i];
	        int[] preRight = new int[pre.length-i-1];
	        int[] inRight = new int[in.length-i-1];
	        for(int j = 0;j<in.length;j++) 
          {
	            if(j<i) 
              {
	                preLeft[j] = pre[j+1];
	                inLeft[j] = in[j];
	            }
              else if(j>i) 
              {
	                preRight[j-i-1] = pre[j];
	                inRight[j-i-1] = in[j];
	            }
	        }
	        root.left = reConstructBinaryTree(preLeft,inLeft);
	        root.right = reConstructBinaryTree(preRight,inRight);
	        return root;
	    }
}
```
### 5、用两个栈实现队列 
题目描述：用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型

```Java
public class TwoStackToOneQueue 
{
	Stack<Integer> stack1 = new Stack<Integer>();
    Stack<Integer> stack2 = new Stack<Integer>();
    public void push(int node) 
    {
        stack1.push(node);
    }
    public int pop() 
    {
        if(stack2.empty())
        {
            while(!stack1.empty())
            {
                stack2.push(stack1.pop());
            }
            
        }
    	return stack2.pop();
    }
}
```
### 6、旋转数组的最小数字 
题目描述：把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。
输入一个非递减排序的数组的一个旋转，输出旋转数组的最小元素。
例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。

NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。

```Java
public class MinNumberInRotateArray
{
	public int minNumberInRotateArray(int [] array)
    {
        int len = array.length;
    	if(len == 0)
            return 0;
        for(int i=0;i<len-1;i++)
            if(array[i] > array[i+1])
            	return array[i+1];
         return array[0];
    }
}
```
### 7、斐波那契数列 
题目描述：大家都知道斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项

 ```Java
 public class Fibonacci 
{
	public static int Fibonacci(int n) 
    {
		int a = 1;
        int b = 1;
        int result = 0;
        if(n==1 || n==2)
            return 1;
        else
        {
            for(int i=3;i<=n;i++)
            {
            	 result = a + b;
            	 b = a;
            	 a = result;
       		}
           return result;
        }
    }
}

 ```
### 8、跳台阶 
题目描述：一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法。

```Java
public class JumpStep 
{
	public int JumpFloor(int target) 
	{
	    if(target == 1 || target == 2) 
	    {
	    	return target;
	    }
	    // 第一阶和第二阶考虑过了，初始当前台阶为第三阶，向后迭代
	 
	    // 思路：当前台阶的跳法总数=当前台阶后退一阶的台阶的跳法总数+当前台阶后退二阶的台阶的跳法总数
	 
	    int jumpSum = 0;// 当前台阶的跳法总数
	 
	    int jumpSumBackStep1 = 2;// 当前台阶后退一阶的台阶的跳法总数(初始值当前台阶是第3阶)
	 
	    int jumpSumBackStep2 = 1;// 当前台阶后退二阶的台阶的跳法总数(初始值当前台阶是第3阶)

	    for(int i = 3; i <= target; i++) 
	    {
	 
	    	jumpSum= jumpSumBackStep1 + jumpSumBackStep2;
	 
	    	jumpSumBackStep2 = jumpSumBackStep1;// 后退一阶在下一次迭代变为后退两阶
	 
	    	jumpSumBackStep1 = jumpSum;                   // 当前台阶在下一次迭代变为后退一阶
	 
	    }
	    return jumpSum;
	  }
}
```
### 9、变态跳台阶 
题目描述:
一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。
```Java
/*关于本题，前提是n个台阶会有一次n阶的跳法。分析如下:
f(1) = 1
f(2) = f(2-1) + f(2-2)         //f(2-2) 表示2阶一次跳2阶的次数。
f(3) = f(3-1) + f(3-2) + f(3-3) 
...
f(n) = f(n-1) + f(n-2) + f(n-3) + ... + f(n-(n-1)) + f(n-n) 
 
说明： 
1）这里的f(n) 代表的是n个台阶有一次1,2,...n阶的 跳法数。
2）n = 1时，只有1种跳法，f(1) = 1
3) n = 2时，会有两个跳得方式，一次1阶或者2阶，这回归到了问题（1） ，f(2) = f(2-1) + f(2-2) 
4) n = 3时，会有三种跳得方式，1阶、2阶、3阶，
    那么就是第一次跳出1阶后面剩下：f(3-1);第一次跳出2阶，剩下f(3-2)；第一次3阶，那么剩下f(3-3)
    因此结论是f(3) = f(3-1)+f(3-2)+f(3-3)
5) n = n时，会有n中跳的方式，1阶、2阶...n阶，得出结论：
    f(n) = f(n-1)+f(n-2)+...+f(n-(n-1)) + f(n-n) => f(0) + f(1) + f(2) + f(3) + ... + f(n-1)
    
6) 由以上已经是一种结论，但是为了简单，我们可以继续简化：
    f(n-1) = f(0) + f(1)+f(2)+f(3) + ... + f((n-1)-1) = f(0) + f(1) + f(2) + f(3) + ... + f(n-2)
    f(n) = f(0) + f(1) + f(2) + f(3) + ... + f(n-2) + f(n-1) = f(n-1) + f(n-1)
    可以得出：
    f(n) = 2*f(n-1)
    
7) 得出最终结论,在n阶台阶，一次有1、2、...n阶的跳的方式时，总得跳法为：
              | 1       ,(n=0 ) 
f(n) =     | 1       ,(n=1 )
              | 2*f(n-1),(n>=2)*/

public class ChangeJumpStep 
{
	 public int JumpFloorII(int target) 
	    {
	        if(target==1)
	        	return 1;
	        else
	        	return 2*JumpFloorII(target-1);
	    }
}
```

### 10、矩形覆盖 
题目描述
我们可以用 2 × 1 的小矩形横着或者竖着去覆盖更大的矩形。请问用 n 个 2×1 的小矩形无重叠地覆盖一个2×n 的大矩形，总共有多少种方法？
```Java
public class RectangularCoverage 
{
	public int RectCover(int target) 
    {
        if(target == 0)
            return 0;
		else if(target == 1)
            return 1;
        else if(target == 2)
            return 2;
        else
            return RectCover(target -1) + RectCover(target - 2);
    }
}
```

### 11、二进制中1的个数 
题目描述:
输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。
```Java
/*如果一个整数不为0，那么这个整数至少有一位是1。如果我们把这个整数减1，
那么原来处在整数最右边的1就会变为0，原来在1后面的所有的0都会变成1(如果最右边的1后面还有0的话)。
其余所有位将不会受到影响。
举个例子：一个二进制数1100，从右边数起第三位是处于最右边的一个1。减去1后，第三位变成0，
它后面的两位0变成了1，而前面的1保持不变，因此得到的结果是1011.
我们发现减1的结果是把最右边的一个1开始的所有位都取反了。这个时候如果我们再把原来的整数和减去1之后的结果做与运算，
从原来整数最右边一个1那一位开始所有位都会变成0。如1100&1011=1000.也就是说，把一个整数减去1，再和原整数做与运算，
会把该整数最右边一个1变成0.那么一个整数的二进制有多少个1，就可以进行多少次这样的操作。*/
public class CountOfOne 
{
	public int NumberOf1(int n) 
    {
		int count = 0;
        while(n!=0)
        {
            ++count;
            n = n & (n-1);
        }
        return count;
    }
}
```

### 12、数值的整数次方 
题目描述:
给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。
```Java
public class NumIntegerPower 
{
	public static void main(String [] args)
	{
		System.out.println(Power(2,5));
	}
	
	public static double Power(double base, int exponent) 
    {
		int n;
		double result = 1;
		
		if(exponent > 0)
			n = exponent;
		else if(exponent<0)
		{
			if(base == 0)
				throw new RuntimeException();
			n = -exponent;
		}
			
		else 
			return 1;
		
		while(n!=0)
		{
			System.out.println("base为"+base);
			if((n&1) == 1)
			{
				result *= base;
				System.out.println("result为"+result);
			}
			base *= base;
			
			n>>=1;
		}
		
        return exponent>0?result:1/result;
  	}
}

```

### 13、调整数组顺序使奇数位于偶数前面 
题目描述:
输入一个整数数组，实现一个函数来调整该数组中数字的顺序，
使得所有的奇数位于数组的前半部分，所有的偶数位于位于数组的后半部分，
并保证奇数和奇数，偶数和偶数之间的相对位置不变。
```Java
public class OddBeforeEven 
{
	public static void main(String [] args)
	{
		int [] a = {1,7,2,3,4,5};
		reOrderArray(a);
		for(int i=0;i<a.length;i++)
			System.out.print(a[i] + " ");
	}
	
	public static void reOrderArray(int [] array)
    {
        int len = array.length;
        
        for(int i=0;i<len;i++)
        {
        	for(int j=len-1;j>i;j--)
        	{
        		if(array[j]%2 ==1 && array[j-1]%2==0)
        		{
        			int temp = array[j-1];
                	array[j-1] = array[j];
                	array[j] = temp;
        		}
        	}
        }
     }
}
```

### 14、链表中倒数第k个结点 
题目描述:
输入一个链表，输出该链表中倒数第k个结点。
```Java
public class TheKthNodeFromBottom 
{
	
	public class ListNode {
	    int val;
	    ListNode next = null;

	    ListNode(int val) {
	        this.val = val;
	    }
	}
	
	public ListNode FindKthToTail(ListNode head,int k) 
    {
        if(head == null || k<=0)
        {
            return null;
        }
        ListNode pre = head;
        ListNode last = head;
        for(int i=1;i<k;i++)
        {
            if(last.next != null)
                last = last.next;
            else
                return null;
        }
        while(last.next !=null)
            {
             last = last.next;
             pre = pre.next;
        }
        return pre;
    }
}
```
### 15、反转链表 
题目描述:
输入一个链表，反转链表后，输出链表的所有元素。
```Java
public class ReverseLink 
{
	
	public class ListNode {
	    int val;
	    ListNode next = null;

	    ListNode(int val) {
	        this.val = val;
	    }
	}
	
	public ListNode ReverseList(ListNode head)
    {
        if(head == null)
            return null;
		ListNode current = head;
        ListNode reverseHead = null;
        ListNode pre = null;
        ListNode temp = null;
        
        
        while(current != null)
        {
            temp = current.next;
            current.next = pre;
            if(temp == null)
                reverseHead =  current;
            
            pre = current;
            current = temp;
        }
        return reverseHead;
    }
}
```

### 16、合并两个排序的链表 
题目描述:
输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。
```Java
public class MergeSortList 
{
	public class ListNode {
	    int val;
	    ListNode next = null;

	    ListNode(int val) {
	        this.val = val;
	    }
	}
	
	//非递归解法
	public ListNode Merge(ListNode list1,ListNode list2) 
    {
        if(list1 == null)
            return list2;
        if(list2 == null)
            return list1;
        ListNode mergeListHead = null;
        ListNode current = null;
        
        
        while(list1!=null && list2!=null)
        {
            if(list1.val <= list2.val)
            {
                if(mergeListHead == null)
                {
                    mergeListHead = current = list1;
                }
                else
                {
                    current.next = list1;
                    current = current.next;
                }
                list1 = list1.next;
            }
            else
            {
                if(mergeListHead == null)
                {
                    mergeListHead = current = list2;
                }
                else
                {
                    current.next = list2;
                    current = current.next;
                }
                list2 = list2.next;
            }
            
        }
        
        if(list1 == null)
            {
                current.next = list2;
            }
            else
            {
                current.next = list1;
            }
        return mergeListHead;
    }
	
	//递归解法
	public ListNode Merge2(ListNode list1,ListNode list2) {
	       if(list1 == null){
	           return list2;
	       }
	       if(list2 == null){
	           return list1;
	       }
	       if(list1.val <= list2.val){
	           list1.next = Merge2(list1.next, list2);
	           return list1;
	       }else{
	           list2.next = Merge2(list1, list2.next);
	           return list2;
	       }       
	   }
}
```
### 17、树的子结构 
题目描述:
输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）
```Java
public class SubstructureOfTree 
{
	
	public class TreeNode {
	    int val = 0;
	    TreeNode left = null;
	    TreeNode right = null;

	    public TreeNode(int val) {
	        this.val = val;

	    }

	}
	
	public class Solution
	{
	    public boolean HasSubtree(TreeNode root1,TreeNode root2) 
	    {
	        if(root1 == null || root2 == null)
	            return false;
	        return isSubtree(root1,root2) || HasSubtree(root1.left,root2) || HasSubtree(root1.right,root2);
	        
	    }
	    public boolean isSubtree(TreeNode root1,TreeNode root2)
	    {
	     	   if(root2 == null)
	               return  true;
	           if(root1 == null)
	               return false;
	        	if(root1.val == root2.val)
	                return isSubtree(root1.left,root2.left) && isSubtree(root1.right,root2.right);
	        	else
	                return false;
	    }
	}
}
```

### 18、二叉树的镜像 
题目描述
操作给定的二叉树，将其变换为源二叉树的镜像。 

输入描述:

二叉树的镜像定义：源二叉树 

    	    8
    	   /  \
    	  6   10
    	 / \  / \
    	5  7 9 11
    	镜像二叉树
    	    8
    	   /  \
    	  10   6
    	 / \  / \
    	11 9 7  5
```Java
public class BinarytreeMirror 
{
	
	public class TreeNode {
	    int val = 0;
	    TreeNode left = null;
	    TreeNode right = null;

	    public TreeNode(int val) {
	        this.val = val;

	    }

	}
	
	public class Solution 
	{
	    TreeNode temp;
	    public void Mirror(TreeNode root) 
	    {
	        if(root == null)
	            return;
	        temp = root.right;
	        root.right = root.left;
	        root.left = temp;
	        Mirror(root.left);
	        Mirror(root.right);
	    }
	}
}
```
### 19、顺时针打印矩阵 
题目描述:
输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，
例如，如果输入如下矩阵： 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 

则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.
```Java

/*解题思路：1、只有一行或者一列直接打印
	2、不能全部构成环的，即遍历到最后只剩下一行或者一列的单独打印
	3、全部能够构成一个环的四个 for 循环直接搞定*/

public class PrinttheMatrixClockwise
{
	ArrayList<Integer> arr = new ArrayList<Integer>();
	public ArrayList<Integer> printMatrix(int [][] matrix) 
    {
		int m = matrix[0].length;//列数
		int n = matrix.length;   //行数
		int col = m;
		int row = n;
		int totelCount = m*n;
		int x = 0;
		int y = 0;
		int count = 0;
		
		if(n<=1)
		{
			for(int i=0;i<m;i++)
				arr.add(matrix[0][i]);
			return arr;
		}
		if(m<=1)
		{
			for(int i=0;i<n;i++)
				arr.add(matrix[i][0]);
			return arr;
		}
		
		while(count<totelCount)
		{
			//非正方形矩阵遍历之后只剩下一行或者一列
			//只有剩下一行
			if((n-1 == 1) && row!=2)
			{
				for(int i=x;i<=m-x;i++)
                    arr.add(matrix[x][i]);
				return arr;
			}
				
			//只有剩下一列
			if(m-1 == 1 && col!=2)
			{
				
				for(int i=y;i<=n-y;i++)
					arr.add(matrix[i][y]);
				return arr;
			}
			
			for(int i=x;i<m;i++)
				arr.add(matrix[x][i]);
			count+=(m-x);
			
            
			for(int i=y+1;i<n;i++)
				arr.add(matrix[i][m-1]);
			count += (n-y-1);
			
            
			for(int i=m-2;i>=x;i--)
				arr.add(matrix[n-1][i]);
			count+=(m-x-1);
			
            
			for(int i=n-2;i>=y+1;i--) 
                arr.add(matrix[i][y]);
			count+=(n-y-2);
			
			++x;
			++y;
			--m;
			--n;
			
		}
		
		return arr;
    }
}
```

### 20、包含min函数的栈 

### 21、栈的压入、弹出序列 
题目描述:
输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。假设压入栈的所有数字均不相等。
例如序列1,2,3,4,5是某栈的压入顺序，序列4，5,3,2,1是该压栈序列对应的一个弹出序列，
但4,3,5,1,2就不可能是该压栈序列的弹出序列。（注意：这两个序列的长度是相等的）
```Java
public class StackPushInPopUp 
{
	public static void main(String [] args)
	{
		int [] a = {1,2,3,4,5};
		int [] b = {4,5,3,2,1};
		System.out.println(IsPopOrder(a,b));
	}
	
	public static boolean IsPopOrder(int [] pushA,int [] popA) 
    {
		if(pushA.length == 0 || popA.length == 0)
			return false;
		
		Stack<Integer> stack = new Stack<Integer>();
		int j = 0;
		for(int i=0;i<pushA.length;i++)
		{
			System.out.println("out"+j+"------------->"+i);
			stack.push(pushA[i]);
			while(j<=i && stack.peek() == popA[j])
			{
				stack.pop();
				j++;
				System.out.println("in"+j);
			}
		}
		
		return stack.isEmpty();
    }
}

```

### 22、从上往下打印二叉树 
题目描述:
从上往下打印出二叉树的每个节点，同层节点从左至右打印。
```Java
//广度优先搜索
public class PrintsBinaryTreeFromTopToBottom 
{
	public class TreeNode 
	{
	    int val = 0;
	    TreeNode left = null;
	    TreeNode right = null;

	    public TreeNode(int val) 
	    {
	        this.val = val;

	    }
	}
	
	ArrayList<Integer> arr = new ArrayList<Integer>();
	Queue<TreeNode> queue = new LinkedList<>();
	public ArrayList<Integer> PrintFromTopToBottom(TreeNode root) 
	{
	    if(root == null)
	    	return arr;
	    queue.offer(root);
        while(!queue.isEmpty())
        {
        	TreeNode node = queue.poll();
        	if(node.left != null)
        		queue.offer(node.left);
        	if(node.right != null)
        		queue.offer(node.right);
        	arr.add(node.val);
        }
	    return arr;
	    
	}
}
```

### 23、二叉搜索树的后序遍历序列 
题目描述:
输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。
如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同。
```Java
public class PostorderOfBinarySearchTree 
{
	/*知识点
	 * 
	 * BST的后序序列的合法序列是，对于一个序列S，最后一个元素是x （也就是根），
	 * 如果去掉最后一个元素的序列为T，那么T满足：
	 * T可以分成两段，前一段（左子树）小于x，
	 * 后一段（右子树）大于x，且这两段（子树）都是合法的后序序列
	 * */
	public  boolean VerifySquenceOfBST(int [] sequence) 
    {
		if(sequence.length == 0)
			return false;
        return judge(sequence,0,sequence.length-1);
    }
	
	public  boolean judge(int [] a,int m,int n)
	{
		if(m>n)
			return true;
		int i = n;
		while(i>m && a[i-1]>a[n]) 
			--i;
		for(int j=i-1;j>=m;--j)
			if(a[j] > a[n])
				return false;
		return judge(a,m,i-1) && judge(a, i, n-1);
	}
}

```

### 24、二叉树中和为某一值的路径 
题目描述:
输入一颗二叉树和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。
路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。
```Java
public class BinaryTreeAndTheValueOfAPath 
{
	public class TreeNode 
	{
	    int val = 0;
	    TreeNode left = null;
	    TreeNode right = null;
	    public TreeNode(int val) 
	    {
	        this.val = val;

	    }

	}
	//深度优先搜索	
  ArrayList <ArrayList<Integer>> listTotal = new ArrayList<>();
	ArrayList<Integer> list = new ArrayList<>();
	public ArrayList<ArrayList<Integer>> FindPath(TreeNode root,int target) 
	{
			if(root == null)
				return listTotal;
			list.add(root.val);
			target -= root.val;
			if(target == 0 && root.left == null && root.right == null)
				listTotal.add(new ArrayList<Integer>(list));
			FindPath(root.left, target);
			FindPath(root.right, target);
			list.remove(list.size()-1);
	    	return listTotal;
	}
}
```

### 25、复杂链表的复制 
题目描述：
输入一个复杂链表（每个节点中有节点值，以及两个指针，
一个指向下一个节点，另一个特殊指针指向任意一个节点），
返回结果为复制后复杂链表的head。
（注意，输出结果中请不要返回参数中的节点引用，否则判题程序会直接返回空）
```Java
public class ComplexLinkCopy 
{
	public class RandomListNode 
	{
	    int label;
	    RandomListNode next = null;
	    RandomListNode random = null;

	    RandomListNode(int label) 
	    {
	        this.label = label;
	    }
	}
	
	/*
    1、复制每个节点，如：复制节点A得到A1，将A1插入节点A后面
    2、遍历链表，A1->random = A->random->next;
    3、将链表拆分成原链表和复制后的链表
	 */
	
	 public RandomListNode Clone(RandomListNode pHead)
	 {
		 	if(pHead == null)
		 		return null;
		 	RandomListNode currentNode = pHead;
		 	while(currentNode != null)
		 	{
		 		RandomListNode newNode = new RandomListNode(currentNode.label);
		 		newNode.next = currentNode.next;
		 		currentNode.next = newNode;
		 		currentNode = newNode.next;
		 	}
		 	
		 	currentNode = pHead;
		 	while(currentNode != null)
		 	{
		 		RandomListNode newNode = currentNode.next;
		 		if(currentNode.random != null)
		 		{
		 			newNode.random = currentNode.random.next;
		 		}
		 		currentNode = newNode.next;
		 	}
		 	
		 	RandomListNode resultHead = pHead.next;
		 	RandomListNode temp;
		 	currentNode = pHead;
		 	while(currentNode.next != null)
		 	{
		 		temp = currentNode.next;
		 		currentNode.next = temp.next;
		 		currentNode = temp;
		 	}
	       	return resultHead;
	 }
}
```

### 26、二叉搜索树与双向链表 

### 27、字符串的排列 

### 28、数组中出现次数超过一半的数字 

### 29、最小的K个数 

### 30、连续子数组的最大和 

### 31、整数中1出现的次数（从1到n整数中1出现的次数）

### 32、把数组排成最小的数 

### 33、丑数 

### 34、第一个只出现一次的字符 

### 35、数组中的逆序对 

### 36、两个链表的第一个公共结点 

### 37、数字在排序数组中出现的次数 

### 38、二叉树的深度 

### 39、平衡二叉树 

### 40、数组中只出现一次的数字 

### 41、和为S的连续正数序列 

### 42、和为S的两个数字 
题目描述:
输入一个递增排序的数组和一个数字S，在数组中查找两个数，是的他们的和正好是S，如果有多对数字的和等于S，输出两个数的乘积最小的。 

输出描述:
对应每个测试案例，输出两个数，小的先输出。
```Java
public class AndSTwoNumbers 
{
	public ArrayList<Integer> FindNumbersWithSum(int [] array,int sum) 
    {
		ArrayList<Integer> arr = new ArrayList<Integer>();
		int i = 0;
		int j = array.length-1;
		while(i<j)
		{
			if(array[i] + array[j] == sum)
			{
				arr.add(array[i]);
				arr.add(array[j]);
				return arr;
			}
			while(i<j && array[i] + array[j] < sum) ++i;
			while(i<j && array[i] + array[j] > sum) --j;
		}
        return arr;
    }
}
```

### 43、左旋转字符串 

### 44、翻转单词顺序列 

### 45、孩子们的游戏(圆圈中最后剩下的数) 

### 46、扑克牌顺子 

### 47、求1+2+3+…+n 
题目描述：
求1+2+3+...+n，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。
```Java
public class Summation 
{
	public static void main(String [] args)
	{
		System.out.println(Sum_Solution(4));
	}
	
	public static int Sum_Solution(int n) 
    {
		int sum = n;
		boolean isSum = (n>0) && ((sum += Sum_Solution(n-1))>0);
        return sum;
    }
}
```
### 48、不用加减乘除做加法 
题目描述:
写一个函数，求两个整数之和，要求在函数体内不得使用+、-、*、/四则运算符号。
```Java
public class Sum 
{
	public static void main(String [] args)
	{
		System.out.println(Add(7,3));
	}
	
	public static int Add(int num1,int num2) 
    {
		int result = num1^num2;
		int a;
		int b = (num1 & num2)<<1;
		while(b!= 0)
		{
			a = result;
			result = a^b;
			b = (a&b)<<1;
		}
        return result;
    }
}
```
### 49、把字符串转换成整数 
题目描述:
将一个字符串转换成一个整数，要求不能使用字符串转换整数的库函数。 数值为0或者字符串不是一个合法的数值则返回0 

输入描述:
输入一个字符串,包括数字字母符号,可以为空

输出描述:
如果是合法的数值表达则返回该数字，否则返回0

输入例子:

+2147483647

    1a33
    
输出例子:

2147483647
    0
    
```Java
//备注
/*0~9 是 48~57
a~z 是 97~122
A~Z 是 65~90*/
public class StringToNum 
{
	public int StrToInt(String str)
    {
        if (str.equals("") || str.length() == 0)
            return 0;
        char[] a = str.toCharArray();
        int fuhao = 0;
        if (a[0] == '-')
            fuhao = 1;
        int sum = 0;
        for (int i = fuhao; i < a.length; i++)
        {
            if (a[i] == '+')
                continue;
            if (a[i] < 48 || a[i] > 57)
                return 0;
            sum = sum * 10 + a[i] - 48;
        }
        return fuhao == 0 ? sum : sum * -1;
    }
}
```

### 50、数组中重复的数字 

### 51、构建乘积数组 

### 52、正则表达式匹配 

### 53、表示数值的字符串 

### 54、字符流中第一个不重复的字符 

### 55、链表中环的入口结点 

### 56、二叉树的下一个结点 

### 57、对称的二叉树 

### 58、按之字形顺序打印二叉树 

### 59、把二叉树打印成多行 

### 60、序列化二叉树 

### 61、二叉搜索树的第k个结点 

### 62、数据流中的中位数 

### 63、滑动窗口的最大值 

### 64、机器人的运动范围 

### 65、矩阵中的路径

未完待续 continue 。。。
