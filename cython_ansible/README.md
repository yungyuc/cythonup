
# Usage

On Ubuntu 14.04

```shell
  $ sudo apt-get update
  $ sudo apt-get install ssh
  $ sudo apt-get install ansible
```

```shell
  $ git clone https://github.com/yungyuc/cythonup.git
  $ cd cythonup/cython_ansible

  $ ansible-playbook -K -i inventory playbook.yml

  # If you want to override the user name in the playbook.yml.
  $ ansible-playbook -u <user_name> -K -i inventory playbook.yml
```
