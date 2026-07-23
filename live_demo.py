"""Entry point for the live edge-node replay demo.

Usage: `python live_demo.py` (after a prior `python pipeline.py` run has
written model_pv.json/model_ev.json to outputs/). See evloadshaping/live_demo.py.
"""

from evloadshaping.live_demo import main

if __name__ == "__main__":
    main()
