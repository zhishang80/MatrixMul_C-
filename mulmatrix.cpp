#include <mpi.h>
#include <iostream>
#include <ctime>

using namespace std;
using namespace MPI;

const int M = 123;
const int N = 203;
const int L = 303;

//Matrix multiply a[L][N] x b[N][M] = c[L][M] by master-slave technology
//Matrix a is decomposed by row (L/nmps) and sent to slave processors 
//         the left extra (L%nmps) part left at master processor
//Matric b is sent to slaves with the entirety
//Matrix c is locally allocated (rows x M) at slaves 
//         and entirety at master for receiving the parts from slaves and the extra part at master 

int main(int argc, char *argv[])
{
    Status status;
    int tag1{100}, tag2{101};

    //Initialize MPI communication world
    MPI::Init();

    //Get the numbers of processors in the communication world and each rank number
    const int nmps = COMM_WORLD.Get_size();
    const int rank = COMM_WORLD.Get_rank();

    //Allocate the 2D matrix b
    double **b = new double*[N];
    for (auto ib=0; ib<N; ib++) b[ib] = new double[M];

    //Calculate the devision number rows and extra at each processor
    const int rows = L/nmps;
    const int extra = L%nmps;

    //Perform the MPI computing, master rank = 0
    if (rank == 0)
    {
        //Allocate the 2D matrix a and c
        double **a = new double*[L];
        for (auto ia=0; ia<L; ia++) a[ia] = new double[N];
        double **c = new double*[L];
        for (auto ic=0; ic<L; ic++) c[ic] = new double[M];

        //Generate the random elements for matrix a and b
        srand(time(NULL));
        for (auto ia=0; ia<L; ia++)
            for(auto ja=0; ja<N; ja++)
                a[ia][ja] = rand() % 10 +1.5;        
        for (auto ib=0; ib<N; ib++)
            for(auto jb=0; jb<M; jb++)
                b[ib][jb] = rand() % 20 +2.5;

        //Send and Receive data between master and slaves
        int offset = 0; //offset = rows*(lcrk-1)
        for (auto lcrk=1; lcrk<nmps; lcrk++)
        {
            //Send the whole matrix b to slaves
            COMM_WORLD.Send(&b[0][0], N*M, MPI_DOUBLE, lcrk, tag1);

            //Send the decomposed matrix a to slaves
            COMM_WORLD.Send(&a[rows*(lcrk-1)][0], rows*N, MPI_DOUBLE, lcrk, tag1);
            //COMM_WORLD.Send(&a[offset][0], rows*N, MPI_DOUBLE, lcrk, tag1);

            //Receive the results of matrix c from slaves
            COMM_WORLD.Recv(&c[offset][0], rows*M, MPI_DOUBLE, lcrk, tag2, status);
            offset = offset + rows;
        }

        //Multiply the extra (rows at master + extra) matrix part left at master
        for (auto ic=L-(rows+extra); ic<L; ic++){
            for (auto jc=0; jc<M; jc++){
                c[ic][jc] = 0.0;
                for (auto ja=0; ja<N; ja++){
                    c[ic][jc] += a[ic][ja] * b[ja][jc];
                }
            }
        }
        //Observe the first and last elements of matrix c
        cout<<"rank="<<rank<<" c0="<<c[0][0]<<"; clm="<<c[L-1][M-1]<<endl;

        //Release the heap memory for avoiding the memory leak
        for (auto ia=0; ia<L; ia++) delete[] a[ia];
        delete[] a;
        for (auto ic=0; ic<L; ic++) delete[] c[ic];
        delete[] c;
    }
    else
    {
        //Allocate local matrix a and c
        double **a = new double*[rows];
        for (auto ia=0; ia<rows; ia++) a[ia] = new double[N];
        double **c = new double*[rows];
        for (auto ic=0; ic<rows; ic++) c[ic] = new double[M];

        //Receive matrix b from master
        COMM_WORLD.Recv(&b[0][0], N*M, MPI_DOUBLE, 0, tag1, status);

        //Reveive decomposed matrix a from master to local matrx a
        COMM_WORLD.Recv(&a[0][0], rows*N, MPI_DOUBLE, 0, tag1, status);
        cout<<"rank="<<rank<<" a0="<<a[0][0]<<" b0="<<b[0][0]<<endl;

        //Perform the multiply 
        for (auto ic=0; ic<rows; ic++){
            for (auto jc=0; jc<M; jc++){
                c[ic][jc] = 0.0;
                for (auto ja=0; ja<N; ja++){
                    c[ic][jc] += a[ic][ja] * b[ja][jc];
                }
            }
        }
        
        //Send the local results of matrix c to master
        COMM_WORLD.Send(&c[0][0], rows*M, MPI_DOUBLE, 0, tag2);
        
        //Release the heap memory for avoiding the memory leak
        for (auto ia=0; ia<rows; ia++) delete[] a[ia];
        delete[] a;
        for (auto ic=0; ic<rows; ic++) delete[] c[ic];
        delete[] c;
    }    

    //Release the heap memory for avoiding the memory leak
    for (auto ib=0; ib<N; ib++) delete[] b[ib];
    delete[] b;

    //Finalize the MPI communication world
    MPI::Finalize();
    return 0;
}
