# thread_mcq_sync
- document: thread
- query: Create 3 multiple-choice questions about thread synchronization
- type: mcq   target: 3   produced: 3

## Q1  [easy]
In a multithreaded program, you must implement entry to a critical section so that when the entry condition is false the thread becomes non‑schedulable and relinquishes the CPU until it can later be awakened on exit. Which synchronization mechanism matches this passive‑waiting behavior and is specifically noted as easy to use but costly because the CPU is relinquished?
- A. Use active waiting with continuous testing so the thread repeatedly checks the condition without blocking.
- B. Use a mutex lock so the thread blocks until it can acquire the lock; on exit, one waiting thread can be woken.
- C. Intentionally create a deadlock so that all threads block until one can proceed.
- D. Guarantee starvation‑freedom so that a waiting thread is eventually allowed to enter without using blocking.

**Answer:** B
**Explanation / model answer:** Passive waiting means the thread blocks (becomes non‑schedulable) and relinquishes the CPU until the condition to enter the critical section becomes true (chunk [28]). Locks (mutexes) are explicitly described as belonging to passive‑waiting solutions and as easy to use but costly because the CPU is relinquished (chunk [31]). Therefore, using a mutex lock that blocks the thread until it can enter is the correct mechanism.

## Q2  [medium]
You are designing exclusive access to a shared in‑memory table. The team insists the system must never get stuck with all contenders waiting forever (even if it's not guaranteed which specific thread gets in next), and that the protection applies precisely to the code that touches the table. According to the notes, which progress property must your solution satisfy, and how does that property connect to the code region that updates the table?
- A. Synchronization is required; it defines the region that uses the shared resource, so some thread will eventually enter that synchronization block.
- B. Deadlock‑freedom is required as a prerequisite for mutual‑exclusion solutions; it guarantees that some thread eventually enters the critical section, which is the code that uses the shared resource.
- C. Deadlock‑freedom is required because it directly defines the shared resource rather than the code that uses it.
- D. Synchronization is required; it guarantees that many threads can be in the region using the shared resource simultaneously, so progress is ensured.

**Answer:** B
**Explanation / model answer:** From the properties to seek in a mutual‑exclusion solution (chunk #30), deadlock‑freedom means that if an activity is trying to enter its critical section, then some activity eventually enters its critical section. Mutual exclusion is about critical sections (chunk #30), and the critical section is where the shared resource is actually used (chunk #28: “In Critical Section: usage of the shared resource”). Therefore, deadlock‑freedom ensures that some thread will eventually enter the code region that updates the shared table.

## Q3  [hard]
Your team wrapped updates to a shared in‑memory table with a mutex. In production, some threads block on the lock and a cycle of waits across several threads leaves all of them stuck. According to the notes, which statement best captures the full reasoning from the mechanism you introduced to the kind of problem you’re now seeing?
- A. Switching to the language’s synchronized keyword instead of a mutex would avoid blocking and therefore prevent deadlock; the current issue is not a synchronization topic.
- B. By adding a mutex you implemented mutual exclusion, which defines the critical section that protects the shared resource; locking can block activities and, when waits become circular, this produces a deadlock — a problem studied under synchronization.
- C. Using a binary semaphore (value 1) would not define a critical section and so would eliminate circular waits; what you observe lies outside synchronization.
- D. Using a mutex implements mutual exclusion, which is directly a synchronization concern; the observed stalls are starvation rather than deadlock and are unrelated to protecting a shared resource.

**Answer:** B
**Explanation / model answer:** Locks (mutex) are used to solve mutual exclusion and belong to passive waiting solutions [31]. Mutual exclusion constrains critical sections — no two activities are in their critical sections at the same time [30] — and the critical section is precisely where the shared resource is used [28]. Because locking can prevent an activity from continuing (blocks it), circular waits among cooperating activities lead to a deadlock [39], which is treated as a synchronization issue in the design notes.
