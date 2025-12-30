# Federal Register Study

**Aim:** Our aim is to help community stakeholders - non-experts but directly affected citizens - engage more easily with policymakers at scale.

**Action:** We're developing a platform that improves Citizen's accessibility to the Federal Register's / Register.gov's commenting process. Recent redesigns improved accessibility; we hope to extend that progress by creating a UX-friendly, modern interface that matches citizens with the policies that impact them and, if they wish, enables them to submit comments directly through our portal.

**Why?:** We believe this work is deeply consequential. Too often, citizens feel excluded from policymaking. The notice-and-comment system is a powerful democratic tool with a strong legal history of protection, but it remains difficult for the public to navigate without domain expertise.

**Alternatives:** A small, concentrated number of politically divisive bills typically a large volume of submissions (on the order of 100,000s), but most fail to receive comments or substantive community comments. Citizens may submit comments directly through the Federal Register or Regulations.

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository and install dependencies
git clone <repo-url>
cd FederalRegisterStudy
uv sync

# Or install in development mode with pip
pip install -e .
```

## Configuration

Set the required environment variables:

```bash
# Regulations.gov API keys (format: KEY1:RPH1,KEY2:RPH2)
export REGS_API_KEYS='your-key-1:5000,your-key-2:1000'

# OpenAI API key for classification
export OPENAI_API_KEY='your-openai-key'
```

## Usage

### Command Line Interface

```bash
# Run full pipeline for a single year
python -m stratification_scripts --years 2024

# Run for multiple years
python -m stratification_scripts --years 2020-2024

# Run specific steps only
python -m stratification_scripts --years 2024 --steps fetch,mine

# Validate environment
python -m stratification_scripts --validate
```

### Shell Wrappers

```bash
# Run single year
./run_single_year.sh 2024

# Run multiple years (default: 2017-2024)
./run_all_years.sh

# With options
./run_single_year.sh 2024 --verbose --limit 100
```

### Programmatic API

```python
from stratification_scripts.pipeline import run_year
from stratification_scripts.config import PipelineConfig

config = PipelineConfig(year=2024, verbose=True)
run_year(config)
```

## Pipeline Steps

The analysis pipeline consists of 4 steps:

1. **fetch** - Fetch Federal Register documents and enrich with Regulations.gov metadata
2. **mine** - Mine comments using stratified sampling from regulations.gov
3. **classify** - Classify comment authors using OpenAI (citizen, org, expert, lobbyist)
4. **plot** - Generate policy-brief visualizations

## Project Structure

```
FederalRegisterStudy/
├── pyproject.toml              # uv/pip dependency management
├── run_all_years.sh            # Multi-year pipeline wrapper
├── run_single_year.sh          # Single-year pipeline wrapper
└── stratification_scripts/
    ├── __init__.py
    ├── cli.py                  # Command-line interface
    ├── pipeline.py             # Pipeline orchestration
    ├── config.py               # Configuration management
    ├── logging_utils.py        # Logging utilities
    ├── io_utils.py             # I/O utilities
    ├── distribution.py         # Document fetching
    ├── openai_client.py        # OpenAI wrapper
    ├── federal_register/       # Federal Register API client
    ├── regulations_gov/        # Regulations.gov API client
    ├── makeup/                 # Comment mining & classification
    │   ├── mine_comments.py
    │   ├── classify_makeup.py
    │   └── data/               # Raw data and results
    └── output/                 # Visualization generation
        └── makeup_plots.py
```

## Our Foci:
We believe there remains huge whitespace in the notice-and-commenting systems in two areas:

*   **Visibility:** Most (important!) notices go unnoticed by ordinary citizens. We match laypeople with bills, notices, and rulechanges that directly affect them or issues they care about.
*   **Ease of Commenting:** Laypeople are often unable or simply too uninterested to pursue long, complex commenting procedures or hash through the nuances of a proper response. We handle the commenting medium itself (ie. via Federal Register, or Regulations.gov, or direct contact) and let people focus on their comments.

Made with love by Praneel Patel [https://praneelp.me/] in collaboration with The Courtyards Institute [https://www.courtyardsinstitute.org/]


**Known Issues:** 
1. Within `stratification_scripts/2024distribution.py`, the sampling will often cap out at 10,000 docs per search. This means not all documents will be collected under a quarterly or longer search timeframe. For our usecase, however, 80,000 document search will suffice as a reasonable porportion.
   UPDATE: FIXED! Searching over days.

2. Data access limitations (counts and channels):
   - What we can access:
     - Federal Register document metadata (titles, dates, agencies, types)
     - Regulations.gov authoritative comment counts for participating agencies (via `commentCount` or `meta.totalElements`)
     - Mappings between FR docs and Regulations.gov (`regs_document_id` when present)
     - Comment period dates when provided
   - What we cannot reliably access:
     - Comment counts for non-participating agency portals (e.g., FCC ECFS, FERC, SEC)
     - Email/mail submission volumes (no centralized API)
     - Counts for agency-specific portals outside Regulations.gov
   - Our approach:
     - Classify submission channels (`regs.gov`, agency portals, email/addresses, other).
     - Compute comment-count percentiles only for the `regs.gov` channel where counts are authoritative; report other channels separately.
