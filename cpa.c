void comunica_vector(double vloc[], int n, int b, int p, double w[]) 
{
    int rank, i;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) 
    {
        for (i=0; i<b; i++) 
        {
            w[i] = vloc[i];
        }
        for (i=1; i<p; i++) 
        {
            MPI_Recv(&w[i * b], b, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        for (i=1; i<p; i++) 
        {
            MPI_Send(w, n, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }
    } else 
    {
        MPI_Send(vloc, b, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        MPI_Recv(w, n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

int rank, i;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
if (rank == 0) 
{
    T1(a, &v);
    MPI_Send(&v, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
    MPI_Recv(&w, 1, MPI_DOUBLE, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    T6(a, &w);
} else if (rank == 1)
{
    T2(b, &w);
    MPI_Send(&w, 1, MPI_DOUBLE, 2, 0, MPI_COMM_WORLD);
    MPI_Recv(&v, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    T3(b, &v);
    MPI_Send(&v, 1, MPI_DOUBLE, 2, 0, MPI_COMM_WORLD)
} else if(rank == 2)
{
    MPI_Recv(&w, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    T4(c, &w);
    MPI_Send(&w, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    MPI_Recv(&v, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    T5(c, &v);
}

void intercambiar(double x[N], int proc1, int proc2) 
{
    int rank, i;
    double xloc[N]
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == proc1)
    {
        MPI_Send(x, N, MPI_DOUBLE, proc2, 0, MPI_COMM_WORLD);
        MPI_Recv(x, N, MPI_DOUBLE, proc2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else if (rank == proc2) 
    {
        MPI_Recv(xloc, N, MPI_DOUBLE, proc1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(x, N, MPI_DOUBLE, proc1, 0, MPI_COMM_WORLD);
        for (i=0; i<N; i++) {
            x[i] = xloc[i];
        }
    }
}

int i, j, rank, k, p;
int A[N][N], B[N][N], v[N], x[N], xloc[N];
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &p);
if (rank == 0) leer(A,v); 
k = N/p;
MPI_Bcast(v, N, MPI_INT, 0, MPI_COMM_WORLD);
MPI_Scatter(A, N*k, MPI_INT, B, N*k, MPI_INT, 0, MPI_COMM_WORLD);
xloc = 0;
for (i=0;i<k;i++) {
    xloc[i]=0;
    for (j=0;j<N;j++) 
    xloc[i] += B[i][j]*v[j];
}
MPI_Gather(xloc, k, MPI_INT, x, k, MPI_INT, 0, MPI_COMM_WORLD);
if (rank == 0) escribir(x);

double infNormPar(double A[][N], double ALocal[][N]) 
{
    int i, j, rank, k, p;
    double s, nrm=0.0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    k = N/p;
    MPI_Scatter(A, N*k, MPI_INT, ALocal, N*k, MPI_INT, 0, MPI_COMM_WORLD);
    for (i=0; i<k; i++) {
        s=0.0;
        for (j=0; j<N; j++)
            s+=fabs(A[i][j]);
        if (s>nrm)
            nrm=s;
    }
    MPI_Reduce(&nrm, 1, , MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
}
t(n)=E[1=0,N-1](E[j=0;N-1](1))=N^2
t(n,p)=k*N+(ta+tw*N*k)*(p-1)+(p-1)*(ta+tw)
void comunica(double A[N][N], double B[N/2][N]) 
{
    int i,j,rank,p;
    MPI_Datatype matriz;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Type_vector(N*N/2, 1, 2, MPI_DOUBLE, &matriz);
    MPI_Type_commit(&matriz);
    if (rank == 0) 
    {
        for (i=1;i<p;i++) 
        {
            if (i%2) 
            {
                MPI_Send(A, 1, matriz, i, 0, MPI_COMM_WORLD);
            } else {
                MPI_Send(&A[1][0], 1, matriz, i, 0, MPI_COMM_WORLD);
            }  
        }
    }
    if (rank%2)
    {
        MPI_Recv(B, N*N/2, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        MPI_Send(B, N*N/2, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_Type_free(&matriz);
}

void distrib(double A[F][C], double Aloc[F][2], MPI_Comm com) 
{
    int i,j,rank,p;
    MPI_Datatype column;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Type_vector(F, 2, C, MPI_DOUBLE, &column);
    MPI_Type_commit(&column);
    if (rank == 0) 
    {
        for(i=1;i<(C/2);i++)
            MPI_Send(&A[0][2], 1, column, i, 0, MPI_COMM_WORLD);
        for(i=0;i<F;i++) {
            Aloc[i][0]=A[i][0];
            Aloc[i][1]=A[i][1];
        }
    } else 
    {
        MPI_Recv(Aloc, 2*F, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_Type_free(&column);
}

void enviof(int x) 
{
    int myid;
    MPI_Comm_rank(&myid);
    if (myid == 0) {
        MPI_Send(&A[x][0], N, MPI_INT, 1, 0, MPI_COMM_WORLD);
    } else {
        MPI_Recv(A, N, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}
void envioc(int x) 
{
    int myid;
    MPI_Datatype column;
    MPI_Comm_rank(&myid);
    MPI_Type_vector(M, 1, N, MPI_INT, &column);
    MPI_Commit(&column);
    if (myid == 0) {
        MPI_Send(&A[0][2], 1, column, 1, 0, MPI_COMM_WORLD);
    } else {
        MPI_Recv(&A[0][2], 1, column, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

double sumaPP(double v[N])
{
int i, myid, p, k;
double s = 0.0, sl = 0.0;
double vloc[N];
MPI_Comm_rank(MPI_COMM_WORLD, &myid);
MPI_Comm_size(MPI_COMM_WORLD, &p);
k=N/p;
if (myid == 0) {
    for (i=1;i<p;i++) {
        MPI_Send(&v[i*k], k, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
    }
    for (i=0; i<k; i++) s += vloc[i];
    for (i=1;i<p;i++) {
        MPI_Recv(&sl, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        s += sl;
    }
    for (i=1;i<p;i++) {
        MPI_Send(&s, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
    }
} else {
    MPI_Recv(vloc, k, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (i=0; i<k; i++) sl += vloc[i];
    MPI_Send(&sl, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    MPI_Recv(&s, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}
return s;
}

double suma(double v[N])
{
int i, myid, p, k;
double s = 0.0, sl = 0.0;
double vloc[N];
MPI_Comm_rank(MPI_COMM_WORLD, &myid);
MPI_Comm_size(MPI_COMM_WORLD, &p);
k=N/p;
MPI_Scatter(v, k, vloc, k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
for (i=0; i<N; i++) sl += vloc[i];
MPI_Reduce(&sl, s, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
MPI_Scatter(&s, 1, s, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
return s;
}

struct Tdatos {
    int x;
    int y[N];
    double a[N];
};

void distribuye(struct Tdatos *datos, int n, MPI_Comm comm) {
    int p, pr;
    MPI_Status status;
    MPI_Datatype tdatos;
    MPI_Aint dir1, dirx, diry, dira;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &pr);
    /*if (pr==0) {
        for (i=1; i<p; i++) {
            MPI_Send(&(datos->x), 1, MPI_INT, i, 0, comm);
            MPI_Send(&(datos->y[0]), n, MPI_INT, i, 0, comm);
            MPI_Send(&(datos->a[0]), n, MPI_DOUBLE, i, 0, comm);
        }
    } else {
        MPI_Recv(&(datos->x), 1, MPI_INT, 0, 0, comm, &status);
        MPI_Recv(&(datos->y[0]), n, MPI_INT, 0, 0, comm, &status);
        MPI_Recv(&(datos->a[0]), n, MPI_DOUBLE, 0, 0, comm, &status);
    }*/
    v={1, n, n};
    MPI_Get_address(datos, &dir1);
    MPI_Get_address(&(datos->x), &dirx);
    MPI_Get_address(&(datos->y[0]), &diry);
    MPI_Get_address(&(datos->a[0]), &dira);
    w={dirx-dir1, diry-dir1, dira-dir1};
    z={MPI_INT, MPI_INT, MPI_DOUBLE};
    MPI_Type_create_struct(3, v, w, z, &tdatos);
    MPI_Type_commit(&tdatos);
    MPI_Bcast(datos, 1, tdatos, 0, comm);
    MPI_Type_free(&tdatos);
}

void func(double A[M][N], int sup[M]) {
    int i, j, p, myid, k, medial;
    double media = 0;
    double Aloc[M][N], 
    int supl[M];
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &myid);
    /* Calcula la media de los elementos de A */
    k=M*N/p;
    MPI_Scatter(A, k, MPI_DOUBLE, Aloc, k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for (i=0; i<k; i++)
        for (j=0; j<N; j++)
            medial += Aloc[i][j];
    MPI_Allreduce(&medial, &media, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    media = media/(M*N);
    /* Cuenta num. de elementos > media en cada fila */
    MPI_Scatter(sup, k, MPI_INT, supl, MPI_INT, 0, MPI_COMM_WORLD);
    for (i=0; i<k; i++) {
        supl[i] = 0;
        for (j=0; j<N; j++)
            if (Aloc[i][j]>media) supl[i]++;
    }
    MPI_Gather(supl, k, MPI_INT, sup, k, MPI_INT, 0, MPI_COMM_WORLD);
}
