import modal
print(dir(modal))
try:
    print(modal.Mount)
except AttributeError:
    print("modal.Mount not found")

try:
    from modal import Mount
    print("from modal import Mount worked")
except ImportError:
    print("from modal import Mount failed")
