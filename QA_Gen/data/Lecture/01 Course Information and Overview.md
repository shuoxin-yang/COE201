# 01 Course Information and Overview

## Course Information
**Instructor:** CHEN Shan 陈杉
**Office:** Room 614, South Tower, CoE Building
**Email:** chens3@sustech.edu.cn

**Q&As:**
* Office hours: 4pm-6pm, Tuesdays, at my office
* QQ group chat (with all TAs) for online Q&A: 907921161

**Platform:**
* Blackboard: bb.sustech.edu.cn ("Discrete Mathematics Fall 2025")
* Please ask questions in class and your classmates will appreciate that! 希望大家课堂上练习用英语交流(不会的部分用中文代替)

## Grading Scheme
| Item | Weight | Notes |
| :--- | :--- | :--- |
| Assignments (~6) | 20% | all submitted in time & each >= 50 points: +1% towards Assignments |
| Quizzes (~2, open-book) | 10% | |
| Midterm (close-book) | 30% | |
| Final (close-book) | 40% | |
| Project (optional) | -1% ~ +5% | |

* We will take random attendance draws in class and each lucky student, if present, gets +1% towards the Assignments & Quizzes (30%) parts.
* **Clarification:** Most material will be covered in class, but some details might be omitted. You are responsible for learning all content in the assigned sections of the textbook, even for those not explicitly taught in class.

## Textbooks & Reference Books
* **Textbook:** Kenneth H. Rosen, *Discrete Mathematics and Its Applications*, Eighth Edition (Mc Graw Hill / 机械工业出版社).
* **Reference Books:**
    * *Concrete Mathematics: A Foundation for Computer Science* by Graham, Knuth, Patashnik.
    * *Introduction to Algorithms*, Second Edition by Cormen, Leiserson, Rivest, Stein.

## Plagiarism Policy
* If plagiarism is found in a student's assignments, course projects, or exams, the corresponding assignment, course project, or exam will receive a score of 0; If the same student is found to have plagiarized for the second time in the same course, the grade for that course will be 0 points.
* If a student does not sign the Declaration Form or cheats in the course, in addition to the grade penalty, the student will not be allowed to enroll in the two CS majors through 1+3 mode, and cannot receive any recommendation for postgraduate admission exam exemption and all other academic awards.
* As it may be difficult to determine who actually wrote it when two assignments are identical or nearly identical, the policy will apply to BOTH students, unless one confesses having copied without the other knowing (uploading your code to public sites like GitHub is considered as one having the knowledge).

**What is OK, and What is not OK?**
* **OK:** Work on an assignment with a friend, and think together about the program/solution structure, share ideas and even the global logic. At the time of actually writing the code or assignment, you should write it alone.
* **OK:** Use in an assignment a piece of code or other resources found on the web, as long as you indicate in a comment/reference where it was found and don't claim it as your own work.
* **OK:** Help friends debug their programs.
* **OK:** Show your code/assignment to friends to explain the logic, as long as the friends write their code/assignment on their own later.
* **NOT OK:** Take the code/assignment of a friend, make a few cosmetic changes (comments, some variable names) and pass it as your own work.

**Assignment 0:**
Please fill out the Undergraduate Students Declaration Form, submit it on Blackboard with your handwritten signature. Otherwise, your course grade will be 0. (You can sign on a tablet, but DO NOT simply type in your name!)

---

## Course Overview

**What is Discrete Mathematics?**
It studies mathematical structures that are discrete (or countable): e.g., integers, graphs, statements in logic, etc. It does not focus on "continuous" mathematics (e.g., real numbers, calculus).
**Why is it important for us?**
It provides an essential foundation for studying and describing objects and problems in almost every area of computer science, since computers operate in "discrete" steps and store data in "discrete" memory.

### Example Topics

**1. Logic and Proofs (logical formulas are discrete structures)**
Translates statements from natural language to symbolic language.
* p: I am interested in Discrete Math
* q: I am taking CS201 this semester
* $p \rightarrow q$ : represents "p implies q"
* premises + axioms + proved theorems $\rightarrow$ conclusion

**2. Number Theory (mainly focusing on integers)**
* Chinese Remainder Theorem (recorded in Sunzi Suanjing):
  $x \equiv 2 \pmod 3$
  $x \equiv 3 \pmod 5$
  $x \equiv 2 \pmod 7$
  How many soldiers in the group? ($x =$ ?)

**3. Recursion (those involving functions defined on integers)**
* Fibonacci sequence:
  $F_0 = 0$, $F_1 = 1$, $\forall n \ge 2 : F_n = F_{n-1} + F_{n-2}$
* Fibonacci spiral / golden spiral (growth factor: golden ratio)
* What is the closed-form expression $F_n =$ ?

**4. Graph Theory ("discrete" model of real-world problems)**
* **Seven Bridges of Königsberg Problem:** People wondered whether it was possible to start at some location in the town, travel across all the bridges once without crossing any bridge twice. In 1736, Leonhard Euler proved this problem has no solution: An Euler walk (traversing each edge once) exists if and only if the graph is connected and it has exactly 0 or 2 nodes of odd degree.
* **The Four-Color Theorem:** Given any separation of a plane into contiguous regions, producing a figure called a map, no more than four colors are required to color the regions of the map so that no two adjacent regions have the same color. In 1976, Kenneth Appel and Wolfgang Haken proved it by cases with a computer—the first well-known computer-aided proof.

**5. Complexity of Algorithms (measuring discrete time and space)**
* **Scheduling Final Exams:** How to schedule the final exams at a university such that no student has two exams at the same time? Vertices represent courses, and there is an edge between two vertices if these courses have a common student. This becomes a graph coloring problem: use min number of colors to color all vertices such that adjacent vertices are colored differently.
* How to measure computational hardness? A literal million-dollar question: $P = NP$?

**6. Combinatorics (counting)**
* What are the odds? (e.g., Odds of winning the Mega Millions jackpot vs. being struck by lightning vs. being attacked by a shark).

**7. Cryptography (often using number theory)**
* How to establish a shared secret key over a public channel (allowing hackers to capture it) without the secret being compromised?

---

## Lecture Schedule (Tentatively)
* Logic and Proofs
* Sets and Functions
* **Quiz 1** (around week 4)
* Complexity of Algorithms
* Number Theory
* Cryptography
* **Midterm Exam**
* Induction and Recursion
* Counting
* **Quiz 2** (around week 12)
* Relations
* Graphs
* Trees
* **Final Exam**

## Learning Objectives
1. Be able to read, understand, and construct mathematical arguments and proofs.
2. Understand the formulations of common problems in several areas of discrete mathematics, including logic and proofs, sets and functions, complexity of algorithms, number theory, cryptography, induction and recursion, counting, relations, graphs and trees, etc.
3. Learn a number of discrete mathematical tools.
4. Apply discrete mathematical tools to solve certain problems in computer science and engineering.