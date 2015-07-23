cython3:
  cmd:
    - run
    - name: pip3 install -U cython
    - require:
      - pkg: python3-pip
