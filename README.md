**Aim:** Our aim is to help community stakeholders - non-experts but directly affected citizens - engage more easily with policymakers at scale.

**Action:** We’re developing a platform that improves Citizen's accessibility to the Federal Register's / Register.gov’s commenting process. Recent redesigns improved accessibility; we hope to extend that progress by creating a UX-friendly, modern interface that matches citizens with the policies that impact them and, if they wish, enables them to submit comments directly through our portal.

**Why?:** We believe this work is deeply consequential. Too often, citizens feel excluded from policymaking. The notice-and-comment system is a powerful democratic tool with a strong legal history of protection, but it remains difficult for the public to navigate without domain expertise.

**Alternatives:** A small, concentrated number of politically divisive bills typically a large volume of submissions (on the order of 100,000s), but most fail to receive comments or substantive community comments. Citizens may submit comments directly through the Federal Register or Regulations.

## Our Foci:
We believe there remains huge whitespace in the notice-and-commenting systems in two areas:

*   **Visibility:** Most (important!) notices go unnoticed by ordinary citizens. We match laypeople with bills, notices, and rulechanges that directly affect them or issues they care about.
*   **Ease of Commenting:** Laypeople are often unable or simply too uninterested to pursue long, complex commenting procedures or hash through the nuances of a proper response. We handle the commenting medium itself (ie. via Federal Register, or Regulations.gov, or direct contact) and let people focus on their comments.

Made with love by Praneel Patel and Emilia Hernandez @ The Courtyards Institute [https://www.courtyardsinstitute.org/]


**Known Issues:** 
1. Within stratification_scripts/2024distribution.py, the sampling will often cap out at 10,000 docs per search. This means not all documents will be collected under a quarterly or longer search timeframe. For our usecase, however, 80,000 document search will suffice as a reasonable porportion. 