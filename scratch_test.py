import meshlib.mrmeshpy as mrmesh

def test_meshlib():
    try:
        help(mrmesh.rayMeshIntersect)
    except Exception as e:
        pass

if __name__ == "__main__":
    test_meshlib()
