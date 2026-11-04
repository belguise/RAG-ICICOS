import os, unittest

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

loader = unittest.TestLoader()
suite = loader.discover('tests', pattern='test_*.py')
runner = unittest.TextTestRunner(verbosity=1)
result = runner.run(suite)

print(f"Ran {result.testsRun} tests. Errors: {len(result.errors)}, Failures: {len(result.failures)}")
if not result.wasSuccessful():
    exit(1)
