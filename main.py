import sys

try:
    from bspline_viewer import BSplineViewer
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure all required modules are installed.")
    sys.exit(1)

def main():
    """Main entry point for the application."""
    try:
        viewer = BSplineViewer()
        viewer.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        import glfw
        glfw.terminate()
        sys.exit(1)

if __name__ == "__main__":
    main()
