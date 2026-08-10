/* tordr.c -- the harness checks its own filing
 *
 * z390 tests a sort with data that narrates itself. APPLE says it is the first
 * record, BANANA says it is the second, and ELDERBERRY is exactly ten
 * characters so it fills the key field with no gap and walks the boundary on
 * its way past. Get the sort wrong and you are not squinting at two columns of
 * numbers, you are reading THIS IS THE THIRD RECORD sitting in second place.
 * The data tells you off itself.
 *
 * Yes, this is cargo-culted from spending too long in z390 and mainframes. I
 * make no apology for it. Every family now leans on the ordering in tmain.c so
 * the ordering gets the same treatment. */

#include "tharns.h"

/* Where a test sits among its own family, once tmain has sorted. */
static int ord_place(const char *me)
{
    int place = 0;
    for (int i = 0; i < th_cnt; i++) {
        if (strcmp(th_list[i].tfam, "ord") != 0) continue;
        place++;
        if (strcmp(th_list[i].tname, me) == 0) return place;
    }
    return -1;
}

/* claim is the position the description spells out in words. */
static int ord_claims(const char *me, int claim, const char *word)
{
    int place = ord_place(me);
    if (place < 0) {
        printf("  %s is not in the listing at all\n", me);
        return 0;
    }
    if (place != claim) {
        printf("  %s SAYS IT IS THE %s RECORD, IT IS SITTING AT %d\n",
               me, word, place);
        return 0;
    }
    return 1;
}

static void ord01(void)
{
    CHECK(ord_claims("ord01", 1, "FIRST"));
    PASS();
}
TH_REG("ord", 1, "THIS IS THE FIRST RECORD BY FAMILY ORDER", ord01)

static void ord02(void)
{
    CHECK(ord_claims("ord02", 2, "SECOND"));
    PASS();
}
TH_REG("ord", 2, "THIS IS THE SECOND RECORD BY FAMILY ORDER", ord02)

static void ord03(void)
{
    CHECK(ord_claims("ord03", 3, "THIRD"));
    PASS();
}
TH_REG("ord", 3, "THIS IS THE THIRD RECORD BY FAMILY ORDER", ord03)

static void ord04(void)
{
    CHECK(ord_claims("ord04", 4, "FOURTH"));
    PASS();
}
TH_REG("ord", 4, "THIS IS THE FOURTH RECORD BY FAMILY ORDER", ord04)

/* ELDERBERRY's job: fill the field exactly, so the boundary gets walked rather
 * than assumed. This description is TH_DESCW characters, which means the dot
 * run between it and the verdict is empty. Retyping it shorter is not a tidy,
 * it is deleting the only test of the wide end of the column. */
static void ord05(void)
{
    const char *me = NULL;
    for (int i = 0; i < th_cnt; i++)
        if (strcmp(th_list[i].tname, "ord05") == 0) me = th_list[i].tdesc;

    CHECK(me != NULL);
    CHEQ((int)strlen(me), TH_DESCW);
    CHECK(ord_claims("ord05", 5, "FIFTH"));
    PASS();
}
TH_REG("ord", 5, "THIS IS THE FIFTH RECORD AND IT FILLS THE LINE", ord05)

/* The families themselves are ordered by the table in tmain.c rather than
 * alphabetically, so smoke runs before the backends. Nothing narrates that,
 * but the sort must at least be total: every test lands in exactly one place
 * and no two tests claim it. */
static void ord06(void)
{
    for (int i = 1; i < th_cnt; i++) {
        if (strcmp(th_list[i].tfam, th_list[i - 1].tfam) != 0) continue;
        if (th_list[i].tnum > th_list[i - 1].tnum) continue;
        printf("  %s follows %s within %s\n",
               th_list[i].tname, th_list[i - 1].tname, th_list[i].tfam);
        nfail++;
        return;
    }
    PASS();
}
TH_REG("ord", 6, "numbers ascend within every family", ord06)

/* A family's tests must be contiguous once sorted, or --fam and the headers
 * disagree about what the family contains. */
static void ord07(void)
{
    for (int i = 0; i < th_cnt; i++) {
        for (int j = i + 2; j < th_cnt; j++) {
            if (strcmp(th_list[i].tfam, th_list[j].tfam) != 0) continue;
            if (strcmp(th_list[j - 1].tfam, th_list[i].tfam) == 0) continue;
            printf("  family %s is split: %s then %s then %s\n",
                   th_list[i].tfam, th_list[i].tname,
                   th_list[j - 1].tname, th_list[j].tname);
            nfail++;
            return;
        }
    }
    PASS();
}
TH_REG("ord", 7, "each family occupies one run", ord07)
