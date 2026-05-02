
# 04 Complexity of Algorithms (算法复杂度)
**Instructor:** Shan Chen | **Course:** CS201 Discrete Mathematics

## 1. Algorithms (算法)
* An **algorithm** is a finite sequence of precise instructions for performing a computation or for solving a problem.

## 2. The Growth of Functions (函数的增长)
In computer science, we are usually interested in what happens when the problem input size $n$ gets big. 
For example, when comparing $n^2/10$ vs $100n+10000$, when $n$ is "large enough" (e.g., $n > 1000$), $n^2/10$ gets bigger and stays bigger for larger $n$.

### Big-O Notation (大 O 记号：上界)
* **Definition:** Let $f$ and $g$ be functions from $\mathbb{Z}$ (or $\mathbb{R}$) to $\mathbb{R}$. We say that $f(x) = O(g(x))$ if there exist positive constants $C$ and $k$ such that:
  $|f(x)| \le C|g(x)|$, whenever $x > k$.
* Big-O gives an **upper bound** on the growth of a function. It tells us that a function grows at most as fast as the other function.
* **Example:** $100n + 10000 = O(n^2/10)$. Note that the opposite is not true: $n^2/10 \ne O(100n + 10000)$.
* Other $O(n^2)$ functions: $4n^2$, $8n^2+2n-3$, $n^2/5+n^{1/2}-10 \log n$, $n(n-3)$.

### Big-O Estimates for Polynomials
* **Theorem:** Let function $f(x) = a_nx^n + a_{n-1}x^{n-1} + \dots + a_1x + a_0$, where $a_0, a_1, \dots, a_n$ ($a_n \ne 0$) are real numbers. Then, $f(x) = O(x^n)$.
* The leading term $a_nx^n$ of a polynomial dominates its growth.
* **Proof:** Assuming $x > 1$, we have:
  $|f(x)| \le |a_n|x^n + |a_{n-1}|x^{n-1} + \dots + |a_1|x + |a_0|$
  $= x^n(|a_n| + |a_{n-1}|/x + \dots + |a_1|/x^{n-1} + |a_0|/x^n)$
  $\le x^n(|a_n| + |a_{n-1}| + \dots + |a_1| + |a_0|)$
  Choose $k=1$ and $C = |a_n| + |a_{n-1}| + \dots + |a_1| + |a_0|$; then $|f(x)| \le Cx^n$ whenever $x > k$.

### Common Big-O Estimates (Order of Growth)
From slowest to fastest growing:
$1 < \log n < n < n \log n < n^2 < 2^n < n!$
* $c = O(1)$
* $\log_a n = O(n)$ for $a > 0$
* $n^a = O(n^b)$ for $0 \le a \le b$
* $cn = O(n)$
* $n^a = O(2^n)$
* $1 + 2 + \dots + n = O(n^2)$
* $\log n! = O(n \log n)$
* $n! = O(n^n)$

### Combination of Functions
* **Theorem (Sum):** If $f_1(x)$ is $O(g_1(x))$ and $f_2(x)$ is $O(g_2(x))$, then $(f_1+f_2)(x) = O(\max(|g_1(x)|, |g_2(x)|))$.
* **Theorem (Product):** If $f_1(x)$ is $O(g_1(x))$ and $f_2(x)$ is $O(g_2(x))$, then $(f_1 f_2)(x) = O(g_1(x) g_2(x))$.

### Big-$\Omega$ Notation (大 $\Omega$ 记号：下界)
* **Definition:** $f(x) = \Omega(g(x))$ if there exist positive constants $C$ and $k$ such that:
  $|f(x)| \ge C|g(x)|$, whenever $x > k$.
* Big-$\Omega$ gives a **lower bound**. It tells us that a function grows at least as fast as the other function.
* **Note:** $f(x) = \Omega(g(x))$ if and only if $g(x) = O(f(x))$.

### Big-$\Theta$ Notation (大 $\Theta$ 记号：紧确界)
* **Definition:** $f(x) = \Theta(g(x))$ if they have the same order of growth: $f(x) = O(g(x))$ and $f(x) = \Omega(g(x))$.

## 3. Complexity of Algorithms
### Problems and Algorithms
* **Computational Problem (计算问题):** A task solved by a computer, formally a set of problem instances together with solutions. 
  * *Example:* Problem: integer factorization. Instance: factor 12. Solution: $12 = 3 \times 4$.
* **Time Complexity:** The number of machine operations (addition, multiplication, assignment, etc.) required by an algorithm.
* **Space Complexity:** The amount of memory used by an algorithm.

### Horner's Method (多项式求值)
* Evaluate $f(x) = a_0 + a_1x + \dots + a_nx^n$
* **Horner's Method:** $f(x) = a_0 + x(a_1 + x(a_2 + \dots + x(a_{n-1} + xa_n)\dots))$
* **Time complexity:** $O(n)$. It takes $n$ multiplications, $n$ additions, and $n$ assignments ($3n$ operations total), which is much better than direct computation.

### Nested Loops Example
```text
S := 0
for i := 1 to n
    for j := 1 to i
        S := S + i * j;
    end for
end for
```
* **Time complexity:** $\Theta(n^2)$. The inner loop runs $1 + 2 + \dots + n = n(n+1)/2$ times.

## 4. Types of Complexity Analysis (以插入排序为例)
Algorithm: **Insertion Sort**
* **Best-Case Complexity:** Fastest possible running time for a given size. Occurs when the array is already sorted: $A[1] \le A[2] \le \dots \le A[n]$. "Key" is compared to only the element right before it. 
  * Time complexity: $\Theta(n)$ ($n-1$ comparisons).
* **Worst-Case Complexity:** Slowest possible running time. Occurs when the array is in reverse order: $A[1] \ge A[2] \ge \dots \ge A[n]$. "Key" is compared to every element before it.
  * Time complexity: $\Theta(n^2)$. ($\sum_{j=2}^{n} (j-1) = n(n-1)/2$ comparisons/swaps).
* **Average-Case Complexity:** Average running time over all possible inputs. On average, "key" is compared to half of the elements before it.
  * Time complexity: $\Theta(n^2)$. ($\sum_{j=2}^{n} (j-1)/2 = n(n-1)/4$ comparisons/swaps).

## 5. Complexity of Problems
### The Input Size of Problems
* Complexity is measured in terms of its **input size** (the number of bits needed to encode the input).
* **Example (COMPOSITE problem):** Given $n$, are there integers $d, k \ge 2$ such that $n = d \times k$? 
  * The input size is the binary representation length: $L = \lceil \log_2(n+1) \rceil \approx \Theta(\log n)$.
  * A naive algorithm checking $d$ from $2$ to $n-1$ takes $\Theta(n)$ divisions. However, relative to input size $L = \log_2 n$, the complexity is $\Theta(2^L)$, which is **exponential** and thus impractical.

### Decision and Optimization Problems
* **Decision Problem (决策问题):** A problem with a yes or no answer (e.g., "Is there an integer $m$ such that $m^m < n$?").
* **Optimization Problem (优化问题):** A problem that asks for optimizing an objective function (e.g., "What is the largest $m$ such that $m^m < n$?").

### Complexity Classes
* **Polynomial-Time Algorithm:** Runs in time $O(n^c)$, where $c>0$ is constant. Usually considered "efficient".
* **Class P (Tractable Problems):** Consists of all decision problems solvable in polynomial time. (e.g., PRIMES).
* **Certificates (证据):** A certificate/proof/witness for a yes-input is a specific object used to verify this input is indeed a yes-input.
* **Class NP (Non-deterministic Polynomial-time):** Consists of all decision problems for which there is a polynomial-time algorithm $V$ that can *verify* a certificate for a yes-input.
  * *Example:* COMPOSITE is in NP because a factor $d$ acts as a certificate, and division takes polynomial time relative to input size.
* **$P = NP$?** One of the most important open problems. We know $P \subseteq NP$, but $NP \subseteq P$ is doubtful because verifying a certificate is usually much easier than finding one.
* **NP-Complete:** The hardest problems in NP. They are polynomial-time reducible to each other. If one has an efficient solution, all do.
* **NP-Hard:** Decision problems that are at least as hard as those in NP-complete (some may not even belong to NP).