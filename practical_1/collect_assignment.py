import os
import argparse

#Files to collect
FILES_TO_COLLECT = ['python_assignment_1.ipynb', 'python_assignment_1.pdf', 
                    'paper_assignment_1_solved.pdf', 'uva_code/__init__.py',
                    'uva_code/layers.py', 'uva_code/losses.py', 'uva_code/models.py',
                    'uva_code/optimizers.py', 'uva_code/solver.py']

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description = 'AssignmentCollector')
  parser.add_argument('--last_name', dest = 'last_name', type = str, default = '')

  args = parser.parse_args()
  last_name = args.last_name
  
  if last_name:
    zip_filename = last_name + '_' + 'assignment_1.zip'
    command = 'zip ' + zip_filename
    for filename in FILES_TO_COLLECT:
      command += ' ' + filename
    os.system(command)
  else:
    raise ValueError("Please, provide your last name in the last_name parameter")


  



