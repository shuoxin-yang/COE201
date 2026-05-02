
# 06 Induction and Recursion
## CS201 Discrete Mathematics
Instructor: Shan Chen

---

### Mathematical Induction

**Principle of Mathematical Induction (数学归纳法原理)**
Let $P(n)$ be a predicate, i.e., $P(n)$ is either true or false for any $n$.
To prove that $P(n)$ is true for all $n \in \mathbb{Z}^+$, we complete two steps:
* **Basis step:** prove $P(1)$ is true
* **Inductive step:** prove $\forall k \in \mathbb{Z}^+, P(k) \rightarrow P(k+1)$ is true

*Note: "$P(k)$ is true" is called the inductive hypothesis (IH) 归纳假设.*

**Why this principle is valid?**
Proof by contradiction: Assume $P(n)$ is false for some integer $n \ge 1$, then the set $S$ of "all positive integer $n$ such that $P(n)$ is false" is not empty. Let $m$ be the smallest integer in $S$. We have $m \ge 2$ as $P(1)$ is true. However, since $P(m-1)$ is true and $P(m-1) \rightarrow P(m)$ is true, $P(m)$ must be true, contradiction!

**Well-Ordering Principle (良序原理)**
Every nonempty subset of $\mathbb{Z}^+$ has a least/minimum element. (This is an axiom).
* This principle is equivalent to mathematical induction.
* This also means mathematical induction can be generalized from $\mathbb{Z}^+$ to any well-ordered set 良序集 $S$, e.g., $\mathbb{N}$, $\{n \in \mathbb{Z} \mid n \ge b\}$, etc.

**Example 1**
Show that $1+2+\cdots+n = n(n+1)/2$ for any positive integer $n$.
*Proof by (mathematical) induction:*
* Let $P(n)$ be the predicate that "the sum of the first $n$ positive integers is equal to $n(n+1)/2$".
* **Basis step:** $P(1)$ is true, because $1 = 1(1+1)/2$.
* **Inductive step:** From the inductive hypothesis, i.e., $P(k)$ is true for an arbitrary positive integer $k$, we need to show that $P(k+1)$ is true, i.e., $1+2+\cdots+k+1 = (k+1)((k+1)+1)/2$.
    $1+2+\cdots+k+(k+1) = k(k+1)/2 + k + 1$
    $= (k(k+1)+2(k+1))/2 = (k+1)(k+2)/2 = (k+1)((k+1)+1)/2$
By mathematical induction, we know that $P(n)$ is true for all positive integers $n$.

**Example 2 (Exercise)**
Prove that for any integer $n \ge 2$, we have $2^{n+1} \ge n^2 + 3$.
*Proof by induction:*
* Let $P(n)$ be $2^{n+1} \ge n^2 + 3$.
* **Basis step:** $P(2)$ is true, because $2^{2+1} = 8 \ge 7 = 2^2 + 3$.
* **Inductive step:** From the inductive hypothesis, i.e., $P(k)$ is true for an arbitrary integer $k \ge 2$, we need to show that $P(k+1)$ is true:
    $2^{(k+1)+1} = 2 \cdot 2^{k+1} \ge 2(k^2+3) = 2k^2+6 = (k+1)^2 - 2k - 1 + k^2 + 6$
    $= (k+1)^2 + (k-1)^2 + 4 \ge (k+1)^2 + 3$
By mathematical induction, $P(n)$ is true for all integers $n \ge 2$.

---

### Strong Induction (强归纳法)

**Second Principle of Mathematical Induction:** To prove that $P(n)$ is true for all $n \in \mathbb{Z}^+$, we complete two steps:
* **Basis step:** prove $P(1)$ is true
* **Inductive step:** prove $\forall k \in \mathbb{Z}^+, P(1) \wedge \cdots \wedge P(k) \rightarrow P(k+1)$ is true

*Note: "$P(1) \wedge P(2) \wedge \cdots \wedge P(k)$ is true" is the inductive hypothesis (IH).*
This is called strong induction or complete induction 完全归纳法. In practice, strong induction is often easier to apply than its weak form, because the inductive hypothesis is stronger. However, these two forms of induction are actually equivalent.

**Example: Prime Factorization**
Theorem: Every positive integer is a power of a prime or the product of powers of primes.
*Proof by strong induction:*
* $P(n)$: "$n$ is a power of a prime or the product of powers of primes"
* **Basis step:** $P(1)$ is true, as 1 is a power of a prime number, $1 = 2^0$.
* **Inductive step:** Inductive hypothesis: $P(m)$ is true for every $m$ that $1 \le m \le k$.
    If $k+1$ is a prime number, $P(k+1)$ is true. Otherwise, $k+1$ must be a composite, i.e., a product of two smaller positive integers, each of which is, by the inductive hypothesis, a power of a prime or the product of powers of primes. Therefore, $P(k+1)$ is true.

---

### Recursion (递归)

**Recursion:** a method of solving a computational problem where its solution depends on solutions to smaller instances of the same problem. Recursive computer programs or algorithms often lead to inductive analysis.

**Example: Towers of Hanoi Puzzle**
Problem: Find an efficient way to move all of the disks from one peg to another, using only legal moves.
* **Basis step:** If $n=1$, moving one disk from one to another is easy.
* **Recursive step:** If $n>1$, we need 3 steps (e.g., to move $n$ disks from peg 1 to peg 3):
    1. Move $n-1$ disks from 1 to 2
    2. Move largest disk from 1 to 3
    3. Move $n-1$ disks from 2 to 3

**Java Implementation:**
```java
public class Hanoi {
    // move n disks from peg a to peg c using peg b
    public void move(int n, char a, char b, char c) {
        if (n == 1) {
            System.out.println("plate " + n + " from " + a + " to " + c);
        } else {
            move(n - 1, a, c, b); // 1. move n-1 disks from a to b using c
            System.out.println("plate " + n + " from " + a + " to " + c); // 2. move the largest disk from a to c
            move(n - 1, b, a, c); // 3. move n-1 disks from b to c using a
        }
    }
}
```

**Towers of Hanoi: Running Time**
Solving the running time (number of disk moves $M(n)$):
* $M(1) = 1$
* $M(n) = 2M(n-1) + 1$ for $n > 1$

Iterating the function gives: $M(1)=1, M(2)=3, M(3)=7, M(4)=15, M(5)=31, \dots$
Guess: $M(n) = 2^n - 1$.
*Proof by induction:*
* **Basis step:** $P(1)$ is true, because $M(1) = 1 = 2^1 - 1$.
* **Inductive step:** Assume $P(k)$ is true for $k \ge 1$, i.e., $M(k) = 2^k - 1$. Then $P(k+1)$ is true: $M(k+1) = 2M(k) + 1 = 2(2^k - 1) + 1 = 2^{k+1} - 1$.

---

### Recurrence Relations (递推关系式)

A recurrence relation tells us how to compute the $n$-th value from some or all of the previous values. To completely specify a function, we must give the initial condition(s) (base cases).

**Example: Number of subsets of a set of size $n$**
$S(n) = \begin{cases} 1 & \text{if } n = 0 \\ 2S(n-1) & \text{if } n \ge 1 \end{cases}$

**Iterating a Recurrence (Backward Substitution)**
Let $T(n) = rT(n-1) + a$.
$T(n) = rT(n-1) + a$
$= r(rT(n-2) + a) + a = r^2T(n-2) + ra + a$
$= r^3T(n-3) + r^2a + ra + a$
$\dots$
$= r^n T(0) + a\sum_{i=0}^{n-1} r^i$

**First-Order Linear Recurrences**
Theorem: For any constants $a$ and $r \neq 0$, and any function $g$, the solution to:
$T(n) = \begin{cases} a & \text{if } n = 0 \\ rT(n-1) + g(n) & \text{if } n > 0 \end{cases}$
is $T(n) = r^n a + \sum_{i=1}^n r^{n-i}g(i)$.

**Closed Formula for $T(n) = rT(n-1) + a$**
If $T(n) = rT(n-1) + a$, $T(0) = b$, and $r \neq 1$, then:
$T(n) = r^n b + a\frac{1-r^n}{1-r}$

**Exercise:**
Solve $T(n) = 4T(n-1) + 2^n \ (n>0)$ with $T(0)=6$.
$T(n) = 6 \cdot 4^n + \sum_{i=1}^n 4^{n-i} \cdot 2^i$
$= 6 \cdot 4^n + 4^n \sum_{i=1}^n 4^{-i} \cdot 2^i = 6 \cdot 4^n + 4^n \sum_{i=1}^n (\frac{1}{2})^i$
$= 6 \cdot 4^n + (1 - \frac{1}{2^n}) \cdot 4^n = 7 \cdot 4^n - 2^n$

---

### Divide-and-Conquer Recurrences (分治)

Divide and Conquer (D&C) recursively breaks down a problem into multiple sub-problems until they are simple enough to be solved directly.

**Example: Binary Search Running Time**
When $n$ is a power of 2:
$T(n) = \begin{cases} 1 & \text{if } n = 1 \\ T(n/2) + 1 & \text{if } n \ge 2 \end{cases}$
Iterating: $T(n) = T(n/2) + 1 = T(n/4) + 2 = \dots = T(n/2^{\log_2 n}) + \log_2 n = 1 + \log_2 n$.

**Iterating D&C Recurrences Examples:**
* $T(n) = 2T(n/2) + n \implies T(n) = nT(1) + n\log_2 n = \Theta(n\log n)$
* $T(n) = T(n/2) + n \implies T(n) = 2n - 1 = \Theta(n)$
* $T(n) = 4T(n/2) + n \implies T(n) = 2n^2 - n = \Theta(n^2)$

**Three Different Behaviors Theorem:**
Consider $T(n) = aT(n/2) + n$ for $n=2^k > 1$, $a \ge 1$, $T(1) = \Theta(1)$.
* If $1 \le a < 2$, then $T(n) = \Theta(n)$.
* If $a = 2$, then $T(n) = \Theta(n \log n)$.
* If $a > 2$, then $T(n) = \Theta(n^{\log_2 a})$.

**Master Theorem (主定理)**
For $T(n) = aT(n/b) + cn^d$ (where $a \ge 1, c > 0, d \ge 0, b \ge 2$):
* If $a < b^d$, then $T(n) = \Theta(n^d)$.
* If $a = b^d$, then $T(n) = \Theta(n^d \log n)$.
* If $a > b^d$, then $T(n) = \Theta(n^{\log_b a})$.
