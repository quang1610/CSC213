#define _XOPEN_SOURCE
#define _XOPEN_SOURCE_EXTENDED

#include "scheduler.h"

#include <assert.h>
#include <curses.h>
#include <ucontext.h>

#include "util.h"

// This is an upper limit on the number of tasks we can create.
#define MAX_TASKS 128

// This is the size of each task's stack memory
#define STACK_SIZE 65536

// These are the possible states of a task
#define RUNNING 0
#define READY 1
#define WAITING_FOR_TASK 2
#define WAITING_FOR_INPUT 3
#define WAITING_FOR_WAKE_UP 4
#define EXITED 5

// This struct will hold the all the necessary information for each task
typedef struct task_info {
  // This field stores all the state required to switch back to this task
  ucontext_t context;

  // This field stores another context. This one is only used when the task
  // is exiting.
  ucontext_t exit_context;

  //   a. Keep track of this task's state.
  //   b. If the task is sleeping, when should it wake up?
  //   c. If the task is waiting for another task, which task is it waiting for?
  //   d. Was the task blocked waiting for user input? Once you successfully
  //      read input, you will need to save it here so it can be returned.
  int state;
  size_t wake_up_time;
  task_t wait_ID;
  int input;

} task_info_t;

int current_task = 0; //< The handle of the currently-executing task
int num_tasks = 1;    //< The number of tasks created so far
int exit_task = 0;
task_info_t tasks[MAX_TASKS]; //< Information for every task

///*************************************************************************************************************
void schedule_new_task() {
  // check the next ready task
  task_t new_task_ID = current_task;
  task_info_t cur_task = tasks[current_task];
  task_info_t new_task;
  int input = 0;

  while (num_tasks != exit_task) {
    printf("current task: %d / new task: %d \n", current_task, new_task_ID);
    new_task = tasks[new_task_ID];

    /// if we found a running task, no need to schedule anything else
    switch (new_task.state) {
    case RUNNING:
    printf("running task %d\n", new_task_ID);
      current_task = new_task_ID;
      return;

    /// if we found a ready task, we will set it to running
    case READY:
      if (new_task_ID != current_task) {
        printf("running task %d\n", new_task_ID);
        new_task.state = RUNNING;
        current_task = new_task_ID;
        swapcontext(&(cur_task.context), &(new_task.context));
        return;
      }
      break;

    /// if we found a task waiting for another one, we will check the another one to see if it exits
    /// if so, we set new task running.
    case WAITING_FOR_TASK:
      if(tasks[new_task.wait_ID].state == EXITED) {
        new_task.state = RUNNING;
        new_task.wait_ID = -1;
        current_task = new_task_ID;
        swapcontext(&(cur_task.context), &(new_task.context));
        return;
      }

      break;

      /// if we found a task waiting for input, we try to read the input
      /// if the input is valid, then we set new_task input and keep it running
    case WAITING_FOR_INPUT:
      input = getch();
      if (getch() != ERR) {
        new_task.state = RUNNING;
        new_task.input = input;
        current_task = new_task_ID;
        swapcontext(&(cur_task.context), &(new_task.context));
        return;
      }
      break;

      /// if we found a sleeping task, we check its wake up time, wake it up if we can and set it running.
    case WAITING_FOR_WAKE_UP:
      if (time_ms() >= new_task.wake_up_time) {
        new_task.state = RUNNING;
        new_task.wake_up_time = 0;
        current_task = new_task_ID;
        swapcontext(&(cur_task.context), &(new_task.context));

        return;
      }
      break;

    default:
      break;
    }

    new_task_ID = (new_task_ID + 1) % num_tasks;
  }
}
///*************************************************************************************************************
/**
 * Initialize the scheduler. Programs should call this before calling any other
 * functiosn in this file.
 */
void scheduler_init() {
  //task_create(0, main);
  tasks[current_task].state = RUNNING;
}

/**
 * This function will execute when a task's function returns. This allows you
 * to update scheduler states and start another task. This function is run
 * because of how the contexts are set up in the task_create function.
 */
void task_exit() {
  tasks[current_task].state = EXITED;
  exit_task += 1;
  schedule_new_task();
}

/**
 * Create a new task and add it to the scheduler.
 *
 * \param handle  The handle for this task will be written to this location.
 * \param fn      The new task will run this function.
 */
void task_create(task_t* handle, task_fn_t fn) {
  // Claim an index for the new task
  int index = num_tasks;
  num_tasks++;

  // Set the task handle to this index, since task_t is just an int
  *handle = index;

  // We're going to make two contexts: one to run the task, and one that runs at the end of the task so we can clean up. Start with the second

  // First, duplicate the current context as a starting point
  getcontext(&tasks[index].exit_context);

  // Set up a stack for the exit context
  tasks[index].exit_context.uc_stack.ss_sp = malloc(STACK_SIZE);
  tasks[index].exit_context.uc_stack.ss_size = STACK_SIZE;

  // Set up a context to run when the task function returns. This should call task_exit.
  makecontext(&tasks[index].exit_context, task_exit, 0);

  // Now we start with the task's actual running context0 / new task: 1
  getcontext(&tasks[index].context);

  // Allocate a stack for the new task and add it to the context
  tasks[index].context.uc_stack.ss_sp = malloc(STACK_SIZE);
  tasks[index].context.uc_stack.ss_size = STACK_SIZE;

  // Now set the uc_link field, which sets things up so our task will go to the exit context when the task function finishes
  tasks[index].context.uc_link = &tasks[index].exit_context;

  // And finally, set up the context to execute the task function
  makecontext(&tasks[index].context, fn, 0);

  tasks[index].state = READY;
  tasks[index].input = ERR;
  tasks[index].wait_ID = -1;
  tasks[index].wake_up_time = 0;
}

/**
 * Wait for a task to finish. If the task has not yet finished, the scheduler should
 * suspend this task and wake it up later when the task specified by handle has exited.
 *
 * \param handle  This is the handle produced by task_create
 */
void task_wait(task_t handle) {
  if(tasks[handle].state != EXITED) {
    tasks[current_task].state = WAITING_FOR_TASK;
    tasks[current_task].wait_ID = handle;
    schedule_new_task();
  }
}

/**
 * The currently-executing task should sleep for a specified time. If that time is larger
 * than zero, the scheduler should suspend this task and run a different task until at least
 * ms milliseconds have elapsed.
 *
 * \param ms  The number of milliseconds the task should sleep.
 */
void task_sleep(size_t ms) {
  size_t wake_up_time = time_ms() + ms;

  tasks[current_task].state = WAITING_FOR_WAKE_UP;
  tasks[current_task].wake_up_time = wake_up_time;

  schedule_new_task();
}

/**
 * Read a character from user input. If no input is available, the task should
 * block until input becomes available. The scheduler should run a different
 * task while this task is blocked.
 *
 * \returns The read character code
 */
int task_readchar() {
  // To check for input, call getch(). If it returns ERR, no input was available.
  // Otherwise, getch() will returns the character code that was read.
  tasks[current_task].state = WAITING_FOR_INPUT;
  schedule_new_task();

  return tasks[current_task].input;
}
