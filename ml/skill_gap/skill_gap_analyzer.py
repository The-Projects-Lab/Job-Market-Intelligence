import pandas as pd


GOLD_PATH = "data/gold_v3/jobs_gold.csv"


class SkillGapAnalyzer:

    def __init__(self):

        self.df = pd.read_csv(
            GOLD_PATH,
            low_memory=False
        )

        self.df["role_category"] = (
            self.df["role_category"]
            .fillna("Unknown")
        )

        self.df["extracted_skills"] = (
            self.df["extracted_skills"]
            .fillna("")
            .str.lower()
        )


    def get_role_skills(
            self,
            role,
            top_n=15
        ):

        role_jobs = self.df[
            self.df["role_category"]
            .str.lower()
            ==
            role.lower()
        ]


        skill_count = {}

        for skills in role_jobs["extracted_skills"]:

            for skill in skills.split(","):

                skill = skill.strip()

                if skill:

                    skill_count[skill] = (
                        skill_count.get(skill,0)
                        +
                        1
                    )


        total_jobs = len(role_jobs)


        result=[]

        for skill,count in skill_count.items():

            result.append(
                {
                "skill":skill,
                "job_count":count,
                "demand_percent":
                    round(
                        count/total_jobs*100,
                        2
                    )
                }
            )


        return (
            pd.DataFrame(result)
            .sort_values(
                "job_count",
                ascending=False
            )
            .head(top_n)
            .reset_index(drop=True)
        )


    def analyze_gap(
            self,
            user_skills,
            target_role
        ):


        required = self.get_role_skills(
            target_role
        )


        user_skills=[
            x.lower().strip()
            for x in user_skills
        ]


        required["status"] = (
            required["skill"]
            .apply(
                lambda x:
                "Available"
                if x in user_skills
                else "Missing"
            )
        )


        available = required[
            required["status"]
            ==
            "Available"
        ]


        missing = required[
            required["status"]
            ==
            "Missing"
        ]


        match_score = round(
            available["demand_percent"].sum()
            /
            required["demand_percent"].sum()
            *
            100,
            2
        )


        learning_path = (
            missing
            .sort_values(
                "demand_percent",
                ascending=False
            )
            ["skill"]
            .head(5)
            .tolist()
        )


        return {

            "role":target_role,

            "match_score":
                match_score,

            "available_skills":
                available,

            "missing_skills":
                missing,

            "learning_path":
                learning_path
        }



if __name__=="__main__":


    analyzer=SkillGapAnalyzer()


    output = analyzer.analyze_gap(

        user_skills=[
            "python",
            "sql",
            "pandas",
            "machine learning"
        ],

        target_role=
        "Data Engineer"
    )


    print(
        "\nROLE:",
        output["role"]
    )


    print(
        "\nPROFILE MATCH:",
        output["match_score"],
        "%"
    )


    print(
        "\nAVAILABLE:"
    )
    print(
        output["available_skills"]
    )


    print(
        "\nMISSING:"
    )
    print(
        output["missing_skills"]
    )


    print(
        "\nRECOMMENDED LEARNING PATH:"
    )

    for i,skill in enumerate(
        output["learning_path"],
        1
    ):
        print(
            i,
            skill
        )
