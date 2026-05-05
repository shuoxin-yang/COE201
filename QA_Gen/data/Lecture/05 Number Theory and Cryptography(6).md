# 05 Number Theory and Cryptography [cite: 1]
## CS201 Discrete Mathematics [cite: 2]
Instructor: Shan Chen [cite: 3]

## ---

### Number Theory [cite: 4]
* Number Theory 数论 is a branch of mathematics that explores integers and their properties. [cite: 5]
* It is the basis of many areas, e.g., cryptography, coding theory, computer security, e-commerce, etc. [cite: 6]
* At one point, the largest employer of mathematicians in the United States, and probably the world, was the National Security Agency (NSA). [cite: 7]
* NSA is the largest spy agency in the US; it is responsible for code design and breaking. [cite: 9, 10]

### Fun Story [cite: 13]
* Godfrey Harold Hardy (1877-1947), UK mathematician. [cite: 14]
* In his autobiography *A Mathematician's Apology*, Hardy wrote: "The great modern achievements of applied mathematics have been in relativity and quantum mechanics, and these subjects are, at present, almost as 'useless' as the theory of numbers." [cite: 15]

## ---

### Divisibility [cite: 20]
* For integers $a$, $b$ with $a \neq 0$, we say that $a$ divides 整除 $b$ if there is an integer $c$ such that $b=ac$, or equivalently $b/a$ is an integer. [cite: 21]
* Here $a$ is a factor or divisor 因数 of $b$, and $b$ is a multiple 倍数 of $a$. [cite: 22]
* Notation: let $a \mid b$ denote $a$ divides $b$ (or $b$ is divisible by $a$) and let $a \nmid b$ denote $a$ does not divide $b$. [cite: 23]
* E.g., we have $4 \mid 24$ and $3 \nmid 17$. [cite: 24]
* All integers divisible by $d>0$ can be enumerated as: $\dots, -kd, \dots, -2d, -d, 0, d, 2d, \dots, kd, \dots$ [cite: 25]
* How many positive integers $\le n$ are divisible by $d>0$? [cite: 26] Count the number of integers written as $kd$ such that $0 < kd \le n$. [cite: 27] Therefore, there are $\lfloor n/d \rfloor$ such positive integers. [cite: 28]

### Divisibility Properties [cite: 31]
* Theorem: Let $a$, $b$, $c$ be integers $(a \neq 0)$. Then [cite: 32]
  * (i) if $a \mid b$ and $a \mid c$, then $a \mid (b+c)$ [cite: 33]
  * (ii) if $a \mid b$ then $a \mid bc$ for all integers $c$ [cite: 34]
  * (iii) if $a \mid b$ and $b \mid c$, then $a \mid c$ [cite: 35]
* Corollary: If $a$, $b$, $c$ are integers $(a \neq 0)$ and $a \mid b$, $a \mid c$ hold, then we have $a \mid (mb+nc)$ for any integers $m$ and $n$. [cite: 37, 38]

### The Division Algorithm [cite: 42]
* Theorem: For any integers $a$, $d$ with $d>0$, there exist unique integers $q$, $r$, with $0 \le r < d$, such that $a = d \cdot q + r$. [cite: 43]
* In this case, $d$ is the divisor 除数, $a$ is the dividend 被除数, $q$ is the quotient 商, and $r$ is the remainder 余数. [cite: 44]
* Notation: $q = a \text{ div } d$, $r = a \bmod d$. [cite: 46]
* Example: $17 = 3 \times 5 + 2 \implies 17 \text{ div } 3 = 5$ and $17 \bmod 3 = 2$. [cite: 48, 49, 50]

### Computing the mod Function [cite: 53]
* Theorem: For integers $a$, $b$, $m$ with $m>0$, we have: [cite: 54]
  * $(a+b) \bmod m = ((a \bmod m) + (b \bmod m)) \bmod m$ [cite: 56]
  * $ab \bmod m = (a \bmod m)(b \bmod m) \bmod m$ [cite: 57]
* Key observation: $a = m(a \text{ div } m) + (a \bmod m) = mq + (a \bmod m)$ [cite: 59, 60]

### Arithmetic Modulo m [cite: 68]
* Let $\mathbb{Z}_m = \{0, 1, \dots, m-1\}$ be the set of nonnegative integers less than $m$. [cite: 69]
* For $a, b \in \mathbb{Z}_m$, addition $+_m$ and multiplication $\cdot_m$ are defined as: [cite: 70]
  * $a +_m b = (a+b) \bmod m$ [cite: 72]
  * $a \cdot_m b = ab \bmod m$ [cite: 76]

### Modular Arithmetic Properties [cite: 83]
* Closure 封闭性: if $a, b \in \mathbb{Z}_m$, then $a +_m b, a \cdot_m b \in \mathbb{Z}_m$ [cite: 84]
* Associativity 结合性: $(a +_m b) +_m c = a +_m (b +_m c)$ and $(a \cdot_m b) \cdot_m c = a \cdot_m (b \cdot_m c)$ [cite: 85, 86]
* Identity Elements 单位元: $a +_m 0 = a$ and $a \cdot_m 1 = a$ [cite: 87]
* Additive Inverses 加法逆: unique inverse $b \in \mathbb{Z}_m$ such that $a +_m b = 0$ (e.g., $m-a$). [cite: 88]
* Commutativity 交换性: $a +_m b = b +_m a$ and $a \cdot_m b = b \cdot_m a$ [cite: 89, 90]
* Distributivity 分配性: $a \cdot_m (b +_m c) = (a \cdot_m b) +_m (a \cdot_m c)$ [cite: 91, 92]

## ---

### Integer Representations [cite: 96, 97]
* Let $b>1$ be an integer. Any positive integer $n$ can be expressed uniquely in the form: [cite: 99]
  $n = a_k b^k + a_{k-1} b^{k-1} + \dots + a_1 b + a_0$ [cite: 100]
  where $k$, $a_i$ are nonnegative integers and $0 \le a_i < b$. [cite: 101]
* This is the base-$b$ expansion $b$进制展开 of $n$, denoted by $(a_k a_{k-1} \dots a_1 a_0)_b$. [cite: 102]

### Constructing Base-b Expansions [cite: 131]
* Algorithm iteratively divides $n$ by $b$ to find remainders ($a_i = q \bmod b$) and updates quotients ($q = q \text{ div } b$) until $q = 0$. [cite: 136, 137, 138]

### Binary Operations of Integers [cite: 148, 166, 181]
* Addition: $O(n) = O(\max(\log a, \log b))$ bit operations. [cite: 151]
* Multiplication: $O(n^2) = O(\log a \log b)$ bit operations. [cite: 168]
* Division (Compute $q = a \text{ div } d$ and $r = a \bmod d$): 
  * Basic method: $\Theta(q \log a)$ operations. [cite: 184]
  * Fast method: $O(\log a \cdot \max(\log q, \log d))$ bit operations recursively using $\lfloor a/2 \rfloor$. [cite: 204, 205]

### Fast Modular Exponentiation [cite: 217]
* Compute $b^n \bmod m$ (where $n = (a_{k-1} \dots a_1 a_0)_2$). [cite: 218]
* $b^n = b^{a_{k-1} 2^{k-1} + \dots + a_1 \cdot 2 + a_0}$ [cite: 220]
* Time complexity: $O(\log b \log m + \log n (\log m)^2)$ bit operations. [cite: 226]

## ---

### Primes and Prime Factorization [cite: 263]
* Prime 素数/质数: a positive integer $p \ge 2$ that has only two positive factors 1 and $p$. [cite: 264]
* Composite 合数: a positive integer $\ge 2$ that is not a prime. [cite: 267]
* Fundamental Theorem of Arithmetic 算术基本定理: Every integer $\ge 2$ can be written uniquely as a prime or as the product of multiple primes in nondecreasing order. [cite: 268]

### Uniqueness of Prime Factorization [cite: 274]
* Lemma: If $p$ is prime and $p \mid a_1 a_2 \dots a_n$, then $p \mid a_i$ for some $i$. [cite: 275]
* Theorem: A prime factorization of a positive integer where primes are in nondecreasing order is unique. [cite: 277]

### Primality Tests [cite: 288]
* Theorem: If $n$ is composite, then $n$ has a prime divisor $\le \sqrt{n}$. [cite: 293]
* Trivial division: test if each prime number $x \le \sqrt{n}$ divides $n$. [cite: 292]
* The Sieve of Eratosthenes is used to find all primes up to a limit by repeatedly deleting multiples of primes. [cite: 301, 303, 318]

### Mersenne Primes [cite: 400]
* Mersenne Prime: a prime of the form $2^p - 1$, where $p$ is prime. [cite: 401]
* Example: $2^5 - 1 = 37$. [cite: 405]
* The largest known prime numbers are Mersenne primes. [cite: 408]

### Conjectures on Primes [cite: 417]
* Goldbach's Conjecture 哥德巴赫猜想 (1+1): Every even integer $> 2$ is the sum of two primes. [cite: 418]
* Twin-Prime Conjecture 孪生素数猜想: There are infinitely many twin primes (primes that differ by 2). [cite: 424, 425]

## ---

### Greatest Common Divisor (GCD) [cite: 428]
* The largest integer $d$ such that $d \mid a$ and $d \mid b$ is the greatest common divisor, $gcd(a,b)$. [cite: 429, 430]
* Found via prime factorization: $gcd(a,b) = p_1^{\min(a_1,b_1)} p_2^{\min(a_2,b_2)} \dots p_n^{\min(a_n,b_n)}$ [cite: 433]
* Two integers $a$ and $b$ are relatively prime 互质 (coprime) if $gcd(a,b) = 1$. [cite: 435]

### Least Common Multiple (LCM) [cite: 438]
* The smallest positive integer divisible by both $a$ and $b$ is the least common multiple, $lcm(a,b)$. [cite: 439]
* Found via prime factorization: $lcm(a,b) = p_1^{\max(a_1,b_1)} \dots p_n^{\max(a_n,b_n)}$ [cite: 444]

### The Euclidean Algorithm [cite: 448]
* Computes GCD efficiently using $a = b \cdot q + r \implies gcd(a,b) = gcd(b,r)$. [cite: 480]
* Iteratively replaces $(a, b)$ with $(b, a \bmod b)$ until $b = 0$. [cite: 464, 465, 466, 467]

### Bézout's Theorem [cite: 507]
* If $a$ and $b$ are positive integers, there exist integers $s$ and $t$ (Bézout coefficients) such that $gcd(a,b) = s \cdot a + t \cdot b$. [cite: 508, 509]
* Corollary: If $gcd(a,b)=1$ and $a \mid bc$, then $a \mid c$. [cite: 558]

## ---

### Congruences [cite: 568]
* Definition: $a \equiv b \pmod m \iff m \mid (a-b)$. [cite: 569]
* Theorem: Let $m$ be a positive integer. If $a \equiv b \pmod m$ and $c \equiv d \pmod m$, then $a+c \equiv b+d \pmod m$ and $ac \equiv bd \pmod m$. [cite: 590]
* Dividing Congruences: If $ac \equiv bc \pmod m$ and $gcd(c,m)=1$, then $a \equiv b \pmod m$. [cite: 605]

### Modular Multiplicative Inverse [cite: 612]
* An integer $\bar{a}$ such that $\bar{a}a \equiv 1 \pmod m$ is the modular multiplicative inverse of $a$ modulo $m$. [cite: 613]
* Theorem: If $gcd(a,m)=1$ and $m>1$, then there exists a unique inverse of $a$ modulo $m$. [cite: 615, 616]
* Found using the Extended Euclidean Algorithm to solve $sa + tm = 1$. [cite: 627, 628]

### Linear Congruences [cite: 644]
* Form: $ax \equiv b \pmod m$. [cite: 645]
* Theorem: Let $d = gcd(a,m)$. The linear congruence has solutions if and only if $d \mid b$. [cite: 666] If so, there are exactly $d$ solutions in $\mathbb{Z}_m$. [cite: 667]

### The Chinese Remainder Theorem [cite: 692]
* Let $m_1, m_2, \dots, m_n$ be pairwise coprime positive integers $\ge 2$. [cite: 693]
* The system $x \equiv a_k \pmod{m_k}$ for $k=1 \dots n$ has a unique solution modulo $M = m_1 m_2 \dots m_n$. [cite: 694, 698]
* Solution: $x = \sum_{k=1}^n a_k M_k y_k$, where $M_k = M/m_k$ and $M_k y_k \equiv 1 \pmod{m_k}$. [cite: 700, 701]

### Fermat's Little Theorem & Euler's Theorem [cite: 737, 750]
* Fermat's Little Theorem: If $p$ is prime and $a \not\equiv 0 \pmod p$, then $a^{p-1} \equiv 1 \pmod p$. [cite: 738, 739]
* Euler's totient function $\phi(n)$ maps $n$ to the number of positive integers coprime to $n$ in $\mathbb{Z}_n$. [cite: 751]
* Euler's Theorem: Let $a, n$ be positive coprime integers. Then $a^{\phi(n)} \equiv 1 \pmod n$. [cite: 759]

### Primitive Roots Modulo a Prime [cite: 764]
* A primitive root 原根 modulo a prime $p$ is an integer $r \in \mathbb{Z}_p$ such that every nonzero element in $\mathbb{Z}_p$ is a power of $r$ modulo $p$. [cite: 765, 766]

## ---

### Hash Functions & PRNGs [cite: 787, 813]
* A hash function maps data of arbitrary length to fixed-length values. Example: $h(k) = k \bmod m$. [cite: 788, 791]
* Pseudorandom numbers are generated by systematic methods. Linear congruential method: $x_{n+1} = (ax_n + c) \bmod m$. [cite: 814, 818]

### Classical Cryptography [cite: 835, 854]
* Shift Cipher / Caesar Cipher: encrypt $m$ as $(m+k) \bmod 26$. [cite: 937, 941]
* Substitution Cipher: Uses a key table mapping each letter. Can be broken by letter frequency analysis. [cite: 948, 949, 954]
* One-Time Pad (OTP): XOR plaintext with a random binary string of the same length. Provides perfect secrecy but has practical drawbacks. [cite: 983, 990, 991, 992]

### Asymmetric / Public-Key Cryptography [cite: 998, 1000]
* Diffie-Hellman (DH) Key Exchange: [cite: 1022]
  * Publicly share prime $p$ and primitive root $g$. [cite: 1025, 1026]
  * Alice computes $A = g^a \bmod p$, Bob computes $B = g^b \bmod p$. [cite: 1031, 1032]
  * Shared secret $K = B^a \bmod p = A^b \bmod p = g^{ab} \bmod p$. [cite: 1033, 1035, 1041]
  * Security based on the Discrete Logarithm Problem (DLP). Insecure against Man-In-The-Middle attacks. [cite: 1046, 1050, 1057]

* The RSA Cryptosystem: [cite: 1078]
  * Pick 2 large primes $p, q$. Let $n=pq$, $\phi(n)=(p-1)(q-1)$. [cite: 1087]
  * Choose $e$ such that $gcd(e, \phi(n)) = 1$ and $d$ such that $ed \equiv 1 \pmod{\phi(n)}$. [cite: 1087]
  * Public key: $(n, e)$. Private key: $d$. [cite: 1091, 1092]
  * Encryption: $c = m^e \bmod n$. [cite: 1088]
  * Decryption: $m = c^d \bmod n$. [cite: 1089]
  * Digital Signature: Sign $s = m^d \bmod n$, Verify $m = s^e \bmod n$. [cite: 1104, 1105]
  * Security is based on the hardness of factoring $n$. [cite: 1094]

### Cryptographic Protocols & Secret Sharing [cite: 1122, 1130]
* Additive Secret Sharing: Dealer shares secret $s$ among $n$ users by distributing $n-1$ random shares $s_i$ and the last share as $s - \sum s_i$. [cite: 1144]
* Shamir's Threshold Secret Sharing: To share a secret $s$ such that $t$ users can reconstruct it, the Dealer picks a random polynomial $f(x)$ with degree $\le t-1$ where $f(0) = s$, and distributes share $s_i = f(i)$ to user $i$. [cite: 1148, 1162]