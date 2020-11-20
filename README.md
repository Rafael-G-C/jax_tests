It is preferable to work in a virtual environment:
```
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```
The virtual environment can be deactivated with:
```
$ deactivate
```

To run the tests:
```
$ python -m pytest -v tests/
```
