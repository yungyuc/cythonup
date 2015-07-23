python3:
  pkg.installed:
    - names:
      - python3
      - python3-dev

pip3:
  pkg.installed:
    - name: python3-pip
