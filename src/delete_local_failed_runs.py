import os
import sys
import shutil

def check_failed_run_and_delete(run_directory):
    log_file = run_directory + os.sep + 'files' + os.sep + 'output.log'
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            lines = f.readlines()
            last_line = lines[-1]
            first_word = last_line.split(':')[0]

            if 'error' in first_word.lower():
                shutil.rmtree(run_directory)
                print(f'Error detected in {run_directory} because of {first_word}. Deleting it.')
                print('Run deleted')
                return True
            else:
                return False
    else:
        return False

print(sys.argv[1])
run_directory = os.path.relpath(sys.argv[1])

deleted_runs = 0
runs = os.listdir(run_directory)
for file_or_directory in runs:
    if os.path.isdir(run_directory + os.sep + file_or_directory):
        print('Checking', file_or_directory)
        deleted = check_failed_run_and_delete(run_directory + os.sep + file_or_directory)
        if deleted:
            deleted_runs += 1

print(f'{deleted_runs} runs deleted')
