import numpy as np
from types import SimpleNamespace
#https://blog.csdn.net/qq_44793283/article/details/145452068
BLACK = 1
WHITE = 0
BLANK = -1
infinity = 100000


class ChessBoard:
    cell = np.full((3, 3), BLANK, dtype=int)
    NowPlayer = BLACK

    def __init__(self):
        pass

    def down_pieces(self, line, column):
        if self.cell[line][column] != BLANK:
            print("落子错误：该位置已有棋子")
            return
        self.cell[line][column] = self.NowPlayer
        self.NowPlayer = not self.NowPlayer
        self.lastLine = line
        self.lastColumn = column

    def back_piece(self, line, column):
        if self.cell[line][column] == -1:
            print("还没下呢，悔什么棋！")
            return
        if self.cell[line][column] == self.NowPlayer:
            print("悔的是别人的棋子！")
            return
        self.cell[line][column] = BLANK
        self.NowPlayer = not self.NowPlayer

    def iswinning(self, player) -> bool:
        # 检查行是否全为该棋 ‌:ml-citation{ref="2,3" data="citationList"}
        row_win = np.any(np.all(self.cell == player, axis=1))
        # 检查列是否全为该棋 ‌:ml-citation{ref="2,3" data="citationList"}
        col_win = np.any(np.all(self.cell == player, axis=0))
        # 检查主对角线是否全为该棋 ‌:ml-citation{ref="1,2" data="citationList"}
        diag1_win = np.all(np.diag(self.cell) == player)
        # 检查副对角线是否全为该棋 ‌:ml-citation{ref="1,2" data="citationList"}
        diag2_win = np.all(np.diag(np.fliplr(self.cell)) == player)
        return row_win or col_win or diag1_win or diag2_win

    def get_blank(self) -> list[tuple[int, int]]:
        # 使用 numpy 查找所有空白格坐标，返回元组列表
        blank_positions = np.argwhere(self.cell == BLANK)
        return [tuple(pos) for pos in blank_positions]

    def evaluate(self):
        if self.iswinning(BLACK):
            return infinity
        elif self.iswinning(WHITE):
            return -infinity
        else:
            if self.cell[1][1] == BLACK:
                return 10  # 中心位置黑棋加分
            elif self.cell[1][1] == WHITE:
                return -10  # 中心位置白棋减分
            else:
                return 0  # 其他情况0分

    def gameIsOver(self):
        if len(self.get_blank()) == 0 or self.iswinning(BLACK) or self.iswinning(WHITE):
            return True
        return False

    def printBoard(self):
        symbol = {1: '  o  ', 0: '  ●  ', -1: '     '}
        print("┌─────┬─────┬─────┐")
        for i in range(3):
            row = [symbol[self.cell[i][j]] for j in range(3)]
            print(f"│{'│'.join(row)}│")
            if i < 2:
                print("├─────┼─────┼─────┤")
        print("└─────┴─────┴─────┘")

    def printBoard1(self):
        print(self.cell)

    def loading(self, file):
        black_count = 0
        white_count = 0
        for i in range(3):
            for j in range(3):
                self.cell[i][j] = file[3 * i + j]
                if file[3 * i + j] == BLACK:
                    black_count += 1
                elif file[3 * i + j] == WHITE:
                    white_count += 1
        if black_count == white_count:
            self.NowPlayer = BLACK
        elif black_count - white_count == 1:
            self.NowPlayer = WHITE
        else:
            print("棋盘加载数据有误")
            return 0
        return 1

    def blank(self):
        if self.get_blank().__len__()==9:
            return True
        return False


def action1(board):
    if board.blank():
        return 1,1
    bestMove = AlthaBeta(board, -infinity, infinity)
    return bestMove.x, bestMove.y


def AlthaBeta(board, altha, beta):
    if board.gameIsOver():
        score = board.evaluate()
        return SimpleNamespace(score=score, x=-1, y=-1)

    if board.NowPlayer == BLACK:
        bestMove = SimpleNamespace(score=-infinity, x=-1, y=-1)
        for (x, y) in board.get_blank():
            board.down_pieces(x, y)
            move = AlthaBeta(board, altha, beta)
            board.back_piece(x, y)
            move.x = x
            move.y = y
            if move.score >= bestMove.score:
                bestMove = move
            altha = bestMove.score
            # if beta <= altha:
                # break
    else:
        bestMove = SimpleNamespace(score=infinity, x=-1, y=-1)
        for (x, y) in board.get_blank():
            board.down_pieces(x, y)
            move = AlthaBeta(board, altha, beta)
            board.back_piece(x, y)
            move.x = x
            move.y = y
            if move.score <= bestMove.score:
                bestMove = move
            beta = bestMove.score
            # if beta <= altha:
                # break
    return bestMove


def action(board):
    if board.gameIsOver():
        if board.iswinning(WHITE):
            print("游戏结束，白棋获胜！")
        if board.iswinning(BLACK):
            print("游戏结束，黑棋获胜！")
        else:
            print("游戏结束，平局！")
    else:
        bestmove = action1(board)
        board.down_pieces(bestmove[0], bestmove[1])
        print(f"下在（{bestmove[0]}，{bestmove[1]}）格子里")
        board.printBoard()
        return bestmove


if __name__ == '__main__':
    board = ChessBoard()
    file = [-1, -1, -1,
            -1, -1, -1,
            -1, -1, -1]
    board.loading(file)
    print(board.blank())
    board.printBoard()
    action(board)
    action(board)
    action(board)
    action(board)
    action(board)


