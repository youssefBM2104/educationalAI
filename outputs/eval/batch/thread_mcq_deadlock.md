# thread_mcq_deadlock
- document: thread
- query: Create 3 multiple-choice questions about deadlock and how it is detected
- type: mcq   target: 3   produced: 3

## Q1  [easy]
In a pthread program, several threads try to enter a region protected by a mutex, but one thread already holds it. According to how this kind of lock behaves, what happens to the other threads, and what event lets one of them proceed?
- A. They engage in active waiting, continuously testing a condition in a loop until they can enter, consuming CPU cycles the entire time.
- B. They perform passive waiting by blocking and relinquishing the CPU; when the owner exits and unlocks, one sleeping thread is woken and can acquire the mutex.
- C. They continue executing inside the protected region in an interleaved fashion, which improves throughput without any blocking.
- D. They are guaranteed starvation-freedom and are time-sliced into the protected region even if the owner never unlocks, so progress occurs without blocking.

**Answer:** B
**Explanation / model answer:** Mutex locks belong to passive waiting solutions: only one thread owns the mutex while others block (become non-schedulable and relinquish the CPU). Upon exit from the critical section, if passive waiting is used, the exiting thread’s unlock wakes one waiting thread so it can acquire the lock. This combines the lock usage sequence (only one owns; owner unlocks; another acquires) with the passive-waiting semantics (block, then wake on exit).

## Q2  [medium]
A team implements the entry to a critical section using passive waiting. When a thread arrives but cannot enter because the section is occupied, what happens to that thread, and what must the exiting thread do to allow progress?
- A. The arriving thread is blocked (temporarily unschedulable), and the exiting thread must explicitly wake one sleeping thread on exit so it can proceed.
- B. Using continuous testing at entry causes the OS to block the thread until exit, so no explicit wake-up is necessary.
- C. The arriving thread spins, repeatedly testing the condition until it becomes true; no wake-up on exit is needed.
- D. Entry simply precedes exit in sequence: the thread goes into the region and later leaves; any wake-up occurs before the exit step.

**Answer:** A
**Explanation / model answer:** From the entry/exit semantics: with passive waiting at entry, a thread that cannot enter becomes non-schedulable (blocked). On exit, if passive waiting is used, one sleeping/waiting thread must be woken so that access can be granted. This behavior is stated explicitly in the entry/exit notes.

## Q3  [hard]
Two processes each need exclusive access to two shared resources and protect each resource with a mutex. Process 1 locks Resource 1 and then tries to lock Resource 2; Process 2 locks Resource 2 and then tries to lock Resource 1. Based on the course notes, what specific outcome occurs and why?
- A. Blocking is avoided because using mutexes as a synchronization technique guarantees that some process will eventually enter its critical section under general design considerations.
- B. No deadlock occurs because mutexes use passive waiting that relinquishes the CPU, allowing the scheduler to run one process and break the cycle.
- C. Deadlock is prevented because Java’s synchronized keyword (an alternative to locks) does not block threads, so both processes continue without waiting.
- D. Both processes block because the mutexes enforce mutual exclusion on critical sections that use the shared resources; each holds one lock while waiting for the other, creating circular blocking that results in a deadlock.

**Answer:** D
**Explanation / model answer:** Locks (mutexes) are used to solve mutual exclusion, ensuring no two activities are in a critical section simultaneously (locks to solve mutual exclusion; mutual exclusion applies to critical sections). In a critical section, a shared resource is being used. The notes emphasize that locking can block an activity’s progress; when two activities each hold one resource and wait for the other, they circularly block, producing a deadlock (requesting a lock can be fatal; circular blocking is deadlock).
