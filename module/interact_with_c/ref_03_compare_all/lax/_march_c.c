void _march_c(double cfl, int nsol, double *sol, double *soln) {
    int itmax = nsol - 1;
    int it = 1;
    while (it < itmax) {
        soln[it] = (1 - cfl*cfl) * sol[it];
        soln[it] += cfl * (cfl+1) / 2 * sol[it-1];
        soln[it] += cfl * (cfl-1) / 2 * sol[it+1];
        it += 1;
    }
    for (it = 0; it < nsol; it++) {
        sol[it] = soln[it];
    }
}

// vim: set ff=unix fenc=utf8 nobomb ai et sw=4 ts=4 tw=79:
