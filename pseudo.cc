int main() {
	if (rank % 2 == 0) {       // FIRST send info THEN recv

		for (int T : timeSteps) {

			// If not rank 0, Send forward (send to rank - 1)
			if (!rank0) {
				Send left edge forntward (send to rank - 1);
			}
			// If not the last rank, Send backward (send to rank + 1)
			if (!the last rank) {
				Send right edge backward (send to rank + 1);
			}

			// Recv from back (recv from rank + 1), if the last rank, get T2
			if (!the last rank) {
				back = recv from back (rank + 1);
			} else {
				back = T2;
			}

			// Recv from front (recv from rank - 1), if rank0 ,get T1
			if (!rank0) {
				front = recv from front (rank - 1);
			} else {
				front = T1;
			}

			// Calculate
			...
			...
			...

			// Update the previous states as current states
			copy(current, current + length, previous);
		}

	} else {    // FIRST recv THEN send

		// Same as above, just send after recv

		for (int T : timeSteps) {
			// Recv from back (recv from rank + 1), if the last rank, get T2
			if (!the last rank) {
				back = recv from back (rank + 1);
			} else {
				back = T2;
			}

			// Recv from front (recv from rank - 1), if rank0 ,get T1
			if (!rank0) {
				front = recv from front (rank - 1);
			} else {
				front = T1;
			}

			// If not rank 0, Send forward (send to rank - 1)
			if (!rank0) {
				Send left edge forntward (send to rank - 1);
			}
			// If not the last rank, Send backward (send to rank + 1)
			if (!the last rank) {
				Send right edge backward (send to rank + 1);
			}


			// Calculate
			...
			...
			...

			// Update the previous states as current states
			copy(current, current + length, previous);
			
		}

	}


	return 0;
}